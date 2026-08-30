use super::*;

/// MAC grid in trunk-local units: x spans [-1, 1] (flame width = 1), height spans
/// [0, GRID_HEIGHT_EXTENT]. Scalars sit on cell centres (index ih * nx + ix),
/// `velocity_x` on the vertical faces ((nx + 1) per row), `velocity_h` on the
/// horizontal faces ((nh + 1) rows of nx).
#[derive(Clone, Debug, PartialEq)]
pub struct GridState {
    pub nx: usize,
    pub nh: usize,
    pub fuel: Vec<f32>,
    pub heat: Vec<f32>,
    pub velocity_x: Vec<f32>,
    pub velocity_h: Vec<f32>,
}

impl GridState {
    pub fn new(nx: usize, nh: usize) -> Self {
        Self {
            nx,
            nh,
            fuel: vec![0.0; nx * nh],
            heat: vec![0.0; nx * nh],
            velocity_x: vec![0.0; (nx + 1) * nh],
            velocity_h: vec![0.0; nx * (nh + 1)],
        }
    }

    pub fn cell_size(&self) -> (f32, f32) {
        (2.0 / self.nx as f32, GRID_HEIGHT_EXTENT / self.nh as f32)
    }

    pub fn cell_centre(&self, ix: usize, ih: usize) -> (f32, f32) {
        let (dx, dh) = self.cell_size();
        (-1.0 + (ix as f32 + 0.5) * dx, (ih as f32 + 0.5) * dh)
    }

    pub fn is_empty(&self) -> bool {
        self.fuel.iter().chain(&self.heat).all(|&v| v == 0.0)
    }

    fn cell_velocity(&self, ix: usize, ih: usize) -> (f32, f32) {
        let row = ih * (self.nx + 1) + ix;
        let column = ih * self.nx + ix;
        (
            0.5 * (self.velocity_x[row] + self.velocity_x[row + 1]),
            0.5 * (self.velocity_h[column] + self.velocity_h[column + self.nx]),
        )
    }

    /// Velocity at a world position, each component sampled on its own face lattice.
    fn velocity_at(&self, x: f32, h: f32) -> (f32, f32) {
        let (dx, dh) = self.cell_size();
        let (sx, sh) = ((x + 1.0) / dx, h / dh);
        (
            sample_cubic(&self.velocity_x, self.nx + 1, self.nh, sx, sh - 0.5),
            sample_cubic(&self.velocity_h, self.nx, self.nh + 1, sx - 0.5, sh),
        )
    }
}

pub fn injection_rate(grid: &FlameGrid, time: f32) -> f32 {
    let pulse = (std::f32::consts::TAU * grid.puff_hz * time).sin();
    (grid.inject_rate * (1.0 + grid.puff_amp.clamp(0.0, 1.0) * pulse)).max(0.0)
}

fn inject(state: &mut GridState, grid: &FlameGrid, time: f32, dt: f32) {
    let rate = injection_rate(grid, time) * dt;
    if rate <= 0.0 {
        return;
    }
    let band = (grid.inject_height * GRID_HEIGHT_EXTENT).max(1e-4);
    let width = grid.inject_width.max(1e-4);
    for ih in 0..state.nh {
        for ix in 0..state.nx {
            let (x, h) = state.cell_centre(ix, ih);
            if h >= band {
                break;
            }
            let profile = (-(x / width) * (x / width)).exp() * (1.0 - h / band);
            let index = ih * state.nx + ix;
            state.fuel[index] += rate * profile;
            state.heat[index] += rate * profile;
        }
    }
}

fn smooth_3x3(field: &[f32], nx: usize, nh: usize) -> Vec<f32> {
    let mut out = field.to_vec();
    for ih in 1..nh - 1 {
        for ix in 1..nx - 1 {
            let mut sum = 0.0;
            for dh in 0..3 {
                for dx in 0..3 {
                    sum += field[(ih + dh - 1) * nx + ix + dx - 1];
                }
            }
            out[ih * nx + ix] = sum / 9.0;
        }
    }
    out
}

fn vorticity(state: &GridState) -> Vec<f32> {
    let (nx, nh) = (state.nx, state.nh);
    let (dx, dh) = state.cell_size();
    let centred: Vec<(f32, f32)> = (0..nx * nh)
        .map(|index| state.cell_velocity(index % nx, index / nx))
        .collect();
    let mut curl = vec![0.0; nx * nh];
    for ih in 1..nh - 1 {
        for ix in 1..nx - 1 {
            let index = ih * nx + ix;
            let duh_dx = (centred[index + 1].1 - centred[index - 1].1) / (2.0 * dx);
            let dux_dh = (centred[index + nx].0 - centred[index - nx].0) / (2.0 * dh);
            curl[index] = duh_dx - dux_dh;
        }
    }
    smooth_3x3(&curl, nx, nh)
}

/// Nguyen 2002 eq. 15 at cell centres: f = epsilon * dx * (N x omega), N the unit gradient of |omega|.
fn confinement_force(state: &GridState, epsilon: f32) -> (Vec<f32>, Vec<f32>) {
    let (nx, nh) = (state.nx, state.nh);
    let (dx, dh) = state.cell_size();
    let curl = vorticity(state);
    let magnitude: Vec<f32> = curl.iter().map(|w| w.abs()).collect();
    let mut force_x = vec![0.0; nx * nh];
    let mut force_h = vec![0.0; nx * nh];
    for ih in 1..nh - 1 {
        for ix in 1..nx - 1 {
            let index = ih * nx + ix;
            let grad_x = (magnitude[index + 1] - magnitude[index - 1]) / (2.0 * dx);
            let grad_h = (magnitude[index + nx] - magnitude[index - nx]) / (2.0 * dh);
            let length = (grad_x * grad_x + grad_h * grad_h).sqrt();
            if length < 1e-6 {
                continue;
            }
            let scale = epsilon * dx * curl[index] / length;
            force_x[index] = scale * grad_h;
            force_h[index] = -scale * grad_x;
        }
    }
    (force_x, force_h)
}

fn apply_forces(state: &mut GridState, grid: &FlameGrid, lateral_gust: f32, dt: f32) {
    let (nx, nh) = (state.nx, state.nh);
    let (force_x, force_h) = if grid.confinement > 0.0 {
        confinement_force(state, grid.confinement)
    } else {
        (vec![0.0; nx * nh], vec![0.0; nx * nh])
    };
    let gust_band = (grid.gust_height * GRID_HEIGHT_EXTENT).max(1e-4);

    for ih in 0..nh {
        let (_, h) = state.cell_centre(0, ih);
        let gust = lateral_gust * (1.0 - h / gust_band).clamp(0.0, 1.0);
        for ix in 0..=nx {
            let left = ih * nx + ix.saturating_sub(1);
            let right = ih * nx + ix.min(nx - 1);
            let confinement = 0.5 * (force_x[left] + force_x[right]);
            state.velocity_x[ih * (nx + 1) + ix] += dt * (gust + confinement);
        }
    }

    for ih in 1..=nh {
        for ix in 0..nx {
            let below = (ih - 1) * nx + ix;
            let above = ih.min(nh - 1) * nx + ix;
            let heat = 0.5 * (state.heat[below] + state.heat[above]);
            let fuel = 0.5 * (state.fuel[below] + state.fuel[above]);
            let buoyancy = grid.buoyancy_heat * heat - grid.buoyancy_density * fuel;
            let confinement = 0.5 * (force_h[below] + force_h[above]);
            state.velocity_h[ih * nx + ix] += dt * (buoyancy + confinement);
        }
    }
    clamp_to_cfl(state, dt);
}

fn clamp_to_cfl(state: &mut GridState, dt: f32) {
    let (dx, dh) = state.cell_size();
    let limit_x = GRID_CFL_LIMIT * dx / dt;
    let limit_h = GRID_CFL_LIMIT * dh / dt;
    for v in &mut state.velocity_x {
        *v = v.clamp(-limit_x, limit_x);
    }
    for v in &mut state.velocity_h {
        *v = v.clamp(-limit_h, limit_h);
    }
}

fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let value = 0.5
        * (2.0 * p1
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t);
    value.clamp(p1.min(p2), p1.max(p2))
}

/// Monotone-clamped bicubic sample of a `width` x `height` lattice at continuous
/// sample coordinates (sample j at coordinate j); outside the lattice reads 0.
pub fn sample_cubic(field: &[f32], width: usize, height: usize, sx: f32, sh: f32) -> f32 {
    if sx < -1.0 || sh < -1.0 || sx > width as f32 || sh > height as f32 {
        return 0.0;
    }
    let (ix, ih) = (sx.floor(), sh.floor());
    let (tx, th) = (sx - ix, sh - ih);
    let at = |x: i64, h: i64| -> f32 {
        if x < 0 || h < 0 || x >= width as i64 || h >= height as i64 {
            0.0
        } else {
            field[h as usize * width + x as usize]
        }
    };
    let (ix, ih) = (ix as i64, ih as i64);
    let rows: Vec<f32> = (-1..=2)
        .map(|k| {
            let h = ih + k;
            catmull_rom(at(ix - 1, h), at(ix, h), at(ix + 1, h), at(ix + 2, h), tx)
        })
        .collect();
    catmull_rom(rows[0], rows[1], rows[2], rows[3], th)
}

/// Semi-Lagrangian advection of one lattice whose sample (i, j) sits at
/// `origin + (i * dx, j * dh)` in world units.
fn advect_lattice(
    field: &[f32],
    width: usize,
    height: usize,
    origin: (f32, f32),
    state: &GridState,
    dt: f32,
) -> Vec<f32> {
    let (dx, dh) = state.cell_size();
    let mut out = vec![0.0; width * height];
    for j in 0..height {
        for i in 0..width {
            let x = origin.0 + i as f32 * dx;
            let h = origin.1 + j as f32 * dh;
            let (vx, vh) = state.velocity_at(x, h);
            let (sx, sh) = ((x - vx * dt - origin.0) / dx, (h - vh * dt - origin.1) / dh);
            out[j * width + i] = sample_cubic(field, width, height, sx, sh);
        }
    }
    out
}

fn advect(state: &mut GridState, dt: f32) {
    let (nx, nh) = (state.nx, state.nh);
    let (dx, dh) = state.cell_size();
    let centre = (-1.0 + 0.5 * dx, 0.5 * dh);
    let velocity_x = advect_lattice(&state.velocity_x, nx + 1, nh, (-1.0, 0.5 * dh), state, dt);
    let velocity_h = advect_lattice(
        &state.velocity_h,
        nx,
        nh + 1,
        (-1.0 + 0.5 * dx, 0.0),
        state,
        dt,
    );
    state.fuel = advect_lattice(&state.fuel, nx, nh, centre, state, dt);
    state.heat = advect_lattice(&state.heat, nx, nh, centre, state, dt);
    state.velocity_x = velocity_x;
    state.velocity_h = velocity_h;
}

fn divergence(state: &GridState) -> Vec<f32> {
    let (nx, nh) = (state.nx, state.nh);
    let (dx, dh) = state.cell_size();
    (0..nx * nh)
        .map(|index| {
            let (ix, ih) = (index % nx, index / nx);
            let row = ih * (nx + 1) + ix;
            (state.velocity_x[row + 1] - state.velocity_x[row]) / dx
                + (state.velocity_h[index + nx] - state.velocity_h[index]) / dh
        })
        .collect()
}

/// Pressure at a neighbour: open sides and top read 0 (free outflow), the floor mirrors (no flow through).
fn pressure_at(pressure: &[f32], nx: usize, nh: usize, ix: i64, ih: i64) -> f32 {
    if ix < 0 || ix >= nx as i64 || ih >= nh as i64 {
        0.0
    } else if ih < 0 {
        pressure[ix as usize]
    } else {
        pressure[ih as usize * nx + ix as usize]
    }
}

/// Red-black Gauss-Seidel on the compact Laplacian; the two colour sweeps keep the result deterministic.
fn solve_pressure(state: &GridState, div: &[f32], iterations: usize) -> Vec<f32> {
    let (nx, nh) = (state.nx, state.nh);
    let (dx, dh) = state.cell_size();
    let (wx, wh) = (1.0 / (dx * dx), 1.0 / (dh * dh));
    let centre = 2.0 * (wx + wh);
    let mut pressure = vec![0.0; nx * nh];
    for _ in 0..iterations {
        for colour in 0..2 {
            for ih in 0..nh as i64 {
                for ix in ((ih as usize + colour) % 2..nx)
                    .step_by(2)
                    .map(|v| v as i64)
                {
                    let neighbours = wx
                        * (pressure_at(&pressure, nx, nh, ix - 1, ih)
                            + pressure_at(&pressure, nx, nh, ix + 1, ih))
                        + wh * (pressure_at(&pressure, nx, nh, ix, ih - 1)
                            + pressure_at(&pressure, nx, nh, ix, ih + 1));
                    let index = ih as usize * nx + ix as usize;
                    pressure[index] = (neighbours - div[index]) / centre;
                }
            }
        }
    }
    pressure
}

fn project(state: &mut GridState, iterations: usize) {
    let (nx, nh) = (state.nx, state.nh);
    let (dx, dh) = state.cell_size();
    let div = divergence(state);
    let pressure = solve_pressure(state, &div, iterations);
    for ih in 0..nh as i64 {
        for ix in 0..=nx as i64 {
            let gradient = (pressure_at(&pressure, nx, nh, ix, ih)
                - pressure_at(&pressure, nx, nh, ix - 1, ih))
                / dx;
            state.velocity_x[ih as usize * (nx + 1) + ix as usize] -= gradient;
        }
    }
    for ih in 1..=nh as i64 {
        for ix in 0..nx as i64 {
            let gradient = (pressure_at(&pressure, nx, nh, ix, ih)
                - pressure_at(&pressure, nx, nh, ix, ih - 1))
                / dh;
            state.velocity_h[ih as usize * nx + ix as usize] -= gradient;
        }
    }
    for ix in 0..nx {
        state.velocity_h[ix] = 0.0;
    }
}

fn decay(state: &mut GridState, grid: &FlameGrid, dt: f32) {
    let burn = (-grid.burn_rate.max(0.0) * dt).exp();
    let cool = (-grid.cool_rate.max(0.0) * dt).exp();
    for v in &mut state.fuel {
        *v = (*v * burn).clamp(0.0, 1.0);
    }
    for v in &mut state.heat {
        *v = (*v * cool).clamp(0.0, 1.0);
    }
}

/// One fixed-dt step: inject -> forces -> advect -> project -> decay (Stam's operator splitting).
/// `lateral_gust` is the root lateral acceleration in x units per second squared at `time`.
pub fn step_grid(state: &mut GridState, grid: &FlameGrid, lateral_gust: f32, time: f32, dt: f32) {
    inject(state, grid, time, dt);
    apply_forces(state, grid, lateral_gust, dt);
    advect(state, dt);
    project(state, grid.pressure_iters.max(0.0) as usize);
    decay(state, grid, dt);
}

pub fn max_divergence(state: &GridState) -> f32 {
    divergence(state)
        .iter()
        .map(|d| d.abs())
        .fold(0.0, f32::max)
}

/// Fuel with a smoothstep fade over the outer GRID_BORDER_FADE_CELLS so the texture border carries no step.
pub fn fuel_texture(state: &GridState) -> Vec<f32> {
    let (nx, nh) = (state.nx, state.nh);
    let fade = |distance: f32| {
        let t = (distance / GRID_BORDER_FADE_CELLS).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    (0..nx * nh)
        .map(|index| {
            let (ix, ih) = ((index % nx) as f32 + 0.5, (index / nx) as f32 + 0.5);
            let edge = ix.min(nx as f32 - ix).min(nh as f32 - ih);
            state.fuel[index] * fade(edge)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 0.04 / 60.0;

    fn grid_params() -> FlameGrid {
        FlameGrid {
            enabled: 1.0,
            ..FlameGrid::default()
        }
    }

    fn gaussian_blob(state: &mut GridState, centre: (f32, f32), sigma_cells: f32) {
        for ih in 0..state.nh {
            for ix in 0..state.nx {
                let dx = ix as f32 + 0.5 - centre.0;
                let dh = ih as f32 + 0.5 - centre.1;
                state.fuel[ih * state.nx + ix] =
                    (-(dx * dx + dh * dh) / (2.0 * sigma_cells * sigma_cells)).exp();
            }
        }
    }

    fn uniform_velocity(state: &mut GridState, cells_per_step_x: f32, cells_per_step_h: f32) {
        let (dx, dh) = state.cell_size();
        state
            .velocity_x
            .iter_mut()
            .for_each(|v| *v = cells_per_step_x * dx / DT);
        state
            .velocity_h
            .iter_mut()
            .for_each(|v| *v = cells_per_step_h * dh / DT);
    }

    fn total(field: &[f32]) -> f32 {
        field.iter().sum()
    }

    fn checkerboard_share(field: &[f32], nx: usize) -> f32 {
        let alternating: f32 = field
            .iter()
            .enumerate()
            .map(|(i, v)| if (i % nx + i / nx) % 2 == 0 { *v } else { -*v })
            .sum();
        alternating.abs() / total(field).max(1e-12)
    }

    #[test]
    fn empty_grid_stays_empty_without_injection() {
        let mut state = GridState::new(GRID_WIDTH_CELLS, GRID_HEIGHT_CELLS);
        let params = FlameGrid {
            inject_rate: 0.0,
            ..grid_params()
        };
        for step in 0..20 {
            step_grid(&mut state, &params, 0.0, step as f32 * DT, DT);
        }
        assert!(state.is_empty());
        assert!(state
            .velocity_x
            .iter()
            .chain(&state.velocity_h)
            .all(|&v| v == 0.0));
    }

    #[test]
    fn projection_removes_divergence() {
        let mut state = GridState::new(GRID_WIDTH_CELLS, GRID_HEIGHT_CELLS);
        let (nx, nh) = (state.nx, state.nh);
        for (index, v) in state.velocity_x.iter_mut().enumerate() {
            let (ix, ih) = ((index % (nx + 1)) as f32, (index / (nx + 1)) as f32);
            *v = (0.3 * ix).sin() * (0.1 * ih).cos();
        }
        for (index, v) in state.velocity_h.iter_mut().enumerate() {
            let (ix, ih) = ((index % nx) as f32, (index / nx) as f32);
            *v = if ih == 0.0 {
                0.0
            } else {
                (0.2 * ih).sin() * (0.15 * ix).cos()
            };
        }
        let before = max_divergence(&state);
        project(&mut state, 3000);
        let after = max_divergence(&state);
        assert!(before > 1.0, "test field must start divergent: {before}");
        assert!(
            after < 1e-3 * before,
            "divergence {after} left from {before}"
        );
        assert!(state.velocity_h[..nx].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn uniform_transport_keeps_mass_within_clamp_loss() {
        let mut state = GridState::new(GRID_WIDTH_CELLS, GRID_HEIGHT_CELLS);
        gaussian_blob(&mut state, (32.0, 30.0), 3.0);
        uniform_velocity(&mut state, 0.3, 0.45);
        let before = total(&state.fuel);
        for _ in 0..40 {
            advect(&mut state, DT);
        }
        let after = total(&state.fuel);
        assert!(
            (after - before).abs() / before < 5e-2,
            "mass {before} -> {after}"
        );
    }

    #[test]
    fn integer_cell_transport_conserves_mass_exactly() {
        let mut state = GridState::new(GRID_WIDTH_CELLS, GRID_HEIGHT_CELLS);
        gaussian_blob(&mut state, (32.0, 30.0), 3.0);
        uniform_velocity(&mut state, 0.0, 1.0);
        let before = total(&state.fuel);
        for _ in 0..40 {
            advect(&mut state, DT);
        }
        assert!((total(&state.fuel) - before).abs() / before < 1e-4);
    }

    #[test]
    fn advected_blob_keeps_its_peak() {
        let mut state = GridState::new(GRID_WIDTH_CELLS, GRID_HEIGHT_CELLS);
        gaussian_blob(&mut state, (32.0, 20.0), 3.0);
        uniform_velocity(&mut state, 0.13, 0.37);
        for _ in 0..100 {
            advect(&mut state, DT);
        }
        let peak = state.fuel.iter().cloned().fold(0.0, f32::max);
        assert!(peak > 0.6, "peak {peak}");
    }

    #[test]
    fn buoyancy_accelerates_heat_linearly() {
        let mut state = GridState::new(GRID_WIDTH_CELLS, GRID_HEIGHT_CELLS);
        let params = FlameGrid {
            confinement: 0.0,
            ..grid_params()
        };
        state.heat.iter_mut().for_each(|v| *v = 0.5);
        for _ in 0..10 {
            apply_forces(&mut state, &params, 0.0, DT);
        }
        let expected = params.buoyancy_heat * 0.5 * 10.0 * DT;
        let face = 60 * state.nx + 32;
        assert!((state.velocity_h[face] - expected).abs() < 1e-5 * expected.max(1.0));
        assert!(state.velocity_x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn injected_column_rises_and_fades() {
        let mut state = GridState::new(GRID_WIDTH_CELLS, GRID_HEIGHT_CELLS);
        let params = grid_params();
        let row_mass =
            |state: &GridState, ih: usize| total(&state.fuel[ih * state.nx..(ih + 1) * state.nx]);
        for step in 0..400 {
            step_grid(&mut state, &params, 0.0, step as f32 * DT, DT);
        }
        assert!(
            row_mass(&state, 30) > 1e-3,
            "fuel never reached the mid column"
        );
        assert!(row_mass(&state, 2) > row_mass(&state, 30));
        assert!(state.velocity_h[30 * state.nx + 32] > 0.0);
        assert!(state.fuel.iter().all(|v| (0.0..=1.0).contains(v)));
        let (_, dh) = state.cell_size();
        let peak_velocity = state.velocity_h.iter().cloned().fold(0.0, f32::max);
        assert!(max_divergence(&state) < 1e-2 * peak_velocity / dh);
    }

    #[test]
    fn confinement_grows_no_checkerboard() {
        let mut state = GridState::new(GRID_WIDTH_CELLS, GRID_HEIGHT_CELLS);
        let params = grid_params();
        for step in 0..300 {
            step_grid(
                &mut state,
                &params,
                4.0 * (step as f32 * 0.3).sin(),
                step as f32 * DT,
                DT,
            );
        }
        assert!(checkerboard_share(&state.fuel, state.nx) < 1e-3);
        assert!(state
            .velocity_x
            .iter()
            .chain(&state.velocity_h)
            .all(|v| v.is_finite()));
    }

    #[test]
    fn steps_are_deterministic() {
        let params = grid_params();
        let run = || {
            let mut state = GridState::new(GRID_WIDTH_CELLS, GRID_HEIGHT_CELLS);
            for step in 0..50 {
                step_grid(&mut state, &params, 2.0, step as f32 * DT, DT);
            }
            state
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn fuel_texture_fades_at_the_border() {
        let mut state = GridState::new(GRID_WIDTH_CELLS, GRID_HEIGHT_CELLS);
        state.fuel.iter_mut().for_each(|v| *v = 1.0);
        let texture = fuel_texture(&state);
        assert!(texture[0] < 0.1);
        assert_eq!(texture[10 * state.nx + 32], 1.0);
        assert!(texture[(state.nh - 1) * state.nx + 32] < 0.1);
        assert_eq!(texture[32], 1.0);
    }
}
