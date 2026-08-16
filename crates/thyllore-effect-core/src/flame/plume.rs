use crate::flame::analytic::radial::evaluate_gaussian_moments;

/// Refractivity of air at STP (dn/dT coefficient).
pub const REFRACTIVITY_AIR: f32 = 2.77e-4;
/// Ambient temperature in Kelvin.
pub const AMBIENT_TEMPERATURE_K: f32 = 293.0;

#[derive(Clone, Copy, Debug)]
pub struct HeatPlume {
    pub plume_temperature: f32,
    pub plume_height: f32,
    pub width_base: f32,
    pub width_slope: f32,
    pub turbulence_amp: f32,
    pub distortion_gain: f32,
}

impl Default for HeatPlume {
    fn default() -> Self {
        Self {
            plume_temperature: 500.0,
            plume_height: 2.0,
            width_base: 0.15,
            width_slope: 0.12,
            turbulence_amp: 0.5,
            distortion_gain: 10.0,
        }
    }
}

/// Plume width at height h.
pub fn plume_width(plume: &HeatPlume, h: f32) -> f32 {
    plume.width_base + plume.width_slope * h.max(0.0)
}

/// Temperature difference at height h.
fn delta_t(plume: &HeatPlume, h: f32) -> f32 {
    let base = plume.plume_temperature - AMBIENT_TEMPERATURE_K;
    let factor = (h + 0.2).powf(-5.0 / 3.0);
    base * factor.min(1.0)
}

/// Refractive index excess at a point p = [x, y, z].
pub fn plume_delta_n(plume: &HeatPlume, p: [f32; 3]) -> f32 {
    let h = p[1];
    let b = plume_width(plume, h);
    let dt = delta_t(plume, h);
    let r2 = p[0] * p[0] + p[2] * p[2];
    -REFRACTIVITY_AIR * (dt / AMBIENT_TEMPERATURE_K) * (-r2 / (b * b)).exp()
}

/// Reference implementation: integrate ∫∇⊥(δn) ds using 4000-step midpoint rule.
/// Returns [dx, dz] deflection.
pub fn plume_deflection_numeric(
    plume: &HeatPlume,
    origin: [f32; 3],
    direction: [f32; 3],
    t_near: f32,
    t_far: f32,
) -> [f32; 2] {
    let steps = 4000;
    let dt = (t_far - t_near) / steps as f32;
    let eps = 1e-3;
    let mut dx = 0.0f32;
    let mut dz = 0.0f32;

    for i in 0..steps {
        let t = t_near + (i as f32 + 0.5) * dt;
        let px = origin[0] + t * direction[0];
        let py = origin[1] + t * direction[1];
        let pz = origin[2] + t * direction[2];

        let p_xp: [f32; 3] = [px + eps, py, pz];
        let p_xm: [f32; 3] = [px - eps, py, pz];
        let p_zp: [f32; 3] = [px, py, pz + eps];
        let p_zm: [f32; 3] = [px, py, pz - eps];

        let dn_xp = plume_delta_n(plume, p_xp);
        let dn_xm = plume_delta_n(plume, p_xm);
        let dn_zp = plume_delta_n(plume, p_zp);
        let dn_zm = plume_delta_n(plume, p_zm);

        dx += (dn_xp - dn_xm) / (2.0 * eps) * dt;
        dz += (dn_zp - dn_zm) / (2.0 * eps) * dt;
    }

    [dx, dz]
}

/// Closed-form implementation using 6 height bands with frozen parameters.
/// Returns [dx, dz] deflection.
pub fn plume_deflection_closed_form(
    plume: &HeatPlume,
    origin: [f32; 3],
    direction: [f32; 3],
    t_near: f32,
    t_far: f32,
) -> [f32; 2] {
    let num_bands = 6;
    let band_dt = (t_far - t_near) / num_bands as f32;
    let mut total_dx = 0.0f32;
    let mut total_dz = 0.0f32;

    for band in 0..num_bands {
        let t0 = t_near + band as f32 * band_dt;
        let t1 = t0 + band_dt;
        let tc = 0.5 * (t0 + t1);

        // Midpoint position
        let pc: [f32; 3] = [
            origin[0] + tc * direction[0],
            origin[1] + tc * direction[1],
            origin[2] + tc * direction[2],
        ];

        let hc = pc[1];
        let b_hc = plume_width(plume, hc);
        let k = 1.0 / (b_hc * b_hc);

        // Gaussian coefficients for integrand (linear in s) × Gaussian
        let a = k * (direction[0] * direction[0] + direction[2] * direction[2]);
        let b_lin = 2.0 * k * (pc[0] * direction[0] + pc[2] * direction[2]);
        let c = k * (pc[0] * pc[0] + pc[2] * pc[2]);

        let half_width = band_dt * 0.5;
        let moments = evaluate_gaussian_moments(a, b_lin, c, half_width);

        // Amplitude
        let amp = -REFRACTIVITY_AIR * (delta_t(plume, hc) / AMBIENT_TEMPERATURE_K);

        // x contribution: -(2k) * amp * (pc[0]*moments[0] + direction[0]*moments[1])
        total_dx += -(2.0 * k) * amp * (pc[0] * moments[0] + direction[0] * moments[1]);

        // z contribution: -(2k) * amp * (pc[2]*moments[0] + direction[2]*moments[1])
        total_dz += -(2.0 * k) * amp * (pc[2] * moments[0] + direction[2] * moments[1]);
    }

    [total_dx, total_dz]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / len, v[1] / len, v[2] / len]
    }

    fn make_ray(ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32) -> ([f32; 3], [f32; 3]) {
        let dir = normalize([dx, dy, dz]);
        ([ox, oy, oz], dir)
    }

    #[test]
    fn test_closed_form_matches_numeric() {
        let plume = HeatPlume::default();
        let rays: [([f32; 3], [f32; 3]); 24] = [
            // Horizontal rays through the plume at various heights
            make_ray(-1.0, 0.5, 0.0, 1.0, 0.0, 0.0),
            make_ray(-1.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            make_ray(-1.0, 1.5, 0.0, 1.0, 0.0, 0.0),
            // Horizontal rays offset in z
            make_ray(-1.0, 0.5, 0.05, 1.0, 0.0, 0.0),
            make_ray(-1.0, 0.5, -0.05, 1.0, 0.0, 0.0),
            // Diagonal rays
            make_ray(-1.0, 0.0, 0.0, 1.0, 0.3, 0.0),
            make_ray(-1.0, 0.0, 0.0, 1.0, -0.3, 0.0),
            make_ray(-1.0, 0.0, 0.0, 1.0, 0.5, 0.0),
            // Rays with z component
            make_ray(-1.0, 0.5, -0.1, 1.0, 0.0, 0.1),
            make_ray(-1.0, 0.5, 0.1, 1.0, 0.0, -0.1),
            // Off-axis rays
            make_ray(-1.0, 0.3, 0.1, 1.0, 0.0, 0.0),
            make_ray(-1.0, 0.7, -0.1, 1.0, 0.0, 0.0),
            // Rays starting inside the plume
            make_ray(0.0, 0.5, 0.0, 1.0, 0.0, 0.0),
            make_ray(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            // Rays at different angles
            make_ray(-1.0, 0.0, 0.0, 1.0, 0.1, 0.05),
            make_ray(-1.0, 0.0, 0.0, 1.0, -0.1, -0.05),
            // Rays grazing the plume edge
            make_ray(-1.0, 0.5, 0.2, 1.0, 0.0, 0.0),
            make_ray(-1.0, 0.5, -0.2, 1.0, 0.0, 0.0),
            // Rays at higher entry points
            make_ray(-1.0, 0.0, 0.0, 1.0, 0.2, 0.0),
            make_ray(-1.0, 0.0, 0.0, 1.0, -0.2, 0.0),
            // Rays with small y component
            make_ray(-1.0, 0.5, 0.0, 1.0, 0.05, 0.0),
            make_ray(-1.0, 0.5, 0.0, 1.0, -0.05, 0.0),
            // Rays passing through center
            make_ray(-1.0, 0.8, 0.0, 1.0, 0.0, 0.0),
            make_ray(-1.0, 0.2, 0.0, 1.0, 0.0, 0.0),
        ];

        let t_near = 0.0;
        let t_far = 2.0;

        let mut max_abs_value: f32 = 0.0;
        for (origin, direction) in &rays {
            let numeric = plume_deflection_numeric(&plume, *origin, *direction, t_near, t_far);
            let closed = plume_deflection_closed_form(&plume, *origin, *direction, t_near, t_far);

            max_abs_value = max_abs_value
                .max(numeric[0].abs())
                .max(numeric[1].abs())
                .max(closed[0].abs())
                .max(closed[1].abs());
        }

        let tolerance = (0.02 * max_abs_value).max(1e-7);

        for (i, (origin, direction)) in rays.iter().enumerate() {
            let numeric = plume_deflection_numeric(&plume, *origin, *direction, t_near, t_far);
            let closed = plume_deflection_closed_form(&plume, *origin, *direction, t_near, t_far);

            let diff_x = (numeric[0] - closed[0]).abs();
            let diff_z = (numeric[1] - closed[1]).abs();

            assert!(
                diff_x < tolerance,
                "Ray {}: x component diff {} >= tolerance {}",
                i,
                diff_x,
                tolerance
            );
            assert!(
                diff_z < tolerance,
                "Ray {}: z component diff {} >= tolerance {}",
                i,
                diff_z,
                tolerance
            );
        }
    }

    #[test]
    fn test_outside_rays_near_zero() {
        let plume = HeatPlume::default();
        let outside_rays: [([f32; 3], [f32; 3]); 4] = [
            // Far above the plume
            make_ray(-1.0, 5.0, 0.0, 1.0, 0.0, 0.0),
            // Far below the plume (negative height)
            make_ray(-1.0, -2.0, 0.0, 1.0, 0.0, 0.0),
            // Far to the side in z
            make_ray(-1.0, 0.5, 5.0, 1.0, 0.0, 0.0),
            // Far to the side in negative z
            make_ray(-1.0, 0.5, -5.0, 1.0, 0.0, 0.0),
        ];

        let t_near = 0.0;
        let t_far = 2.0;

        for (i, (origin, direction)) in outside_rays.iter().enumerate() {
            let numeric = plume_deflection_numeric(&plume, *origin, *direction, t_near, t_far);
            let closed = plume_deflection_closed_form(&plume, *origin, *direction, t_near, t_far);

            assert!(
                numeric[0].abs() < 1e-9 && numeric[1].abs() < 1e-9,
                "Outside ray {}: numeric result [{}, {}], expected near zero",
                i,
                numeric[0],
                numeric[1]
            );
            assert!(
                closed[0].abs() < 1e-9 && closed[1].abs() < 1e-9,
                "Outside ray {}: closed_form result [{}, {}], expected near zero",
                i,
                closed[0],
                closed[1]
            );
        }
    }
}
