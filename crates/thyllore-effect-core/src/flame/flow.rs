use super::*;
use crate::flame::branch::hash01;

const GUST_HARMONICS: [(f32, f32); 3] = [(1.0, 1.0), (1.618, 0.5), (2.414, 0.3)];
const BURST_PERIODS_PER_GUST: f32 = 10.0;
const BURST_WIDTH_OVER_PERIOD: f32 = 0.08;
const PLANE_COUNT: usize = 2;
const FLOW_STEPS_PER_PERIOD: f32 = 40.0;
const FLOW_DAMPING_MEMORY_FOLDS: f32 = 5.0;
const LOBE_SPAWN_JITTER: f32 = 0.5;
const LOBE_HEIGHT_SCATTER: f32 = 0.3;
const LOBE_SIZE_SCATTER: f32 = 0.5;
const LOBE_SEED: u32 = 23;

/// Column markers in flame-local units: lateral centre offset per plane (x, z)
/// and the width scale, one entry per height `i / (FLOW_MARKER_COUNT - 1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowMarker {
    pub offset: [f32; PLANE_COUNT],
    pub width_scale: f32,
}

impl Default for FlowMarker {
    fn default() -> Self {
        Self {
            offset: [0.0; PLANE_COUNT],
            width_scale: 1.0,
        }
    }
}

/// Geometry the simulation runs in: isotropic flame-local units (bounding
/// radius = 1), height01 scaled by `aspect`, base trunk radius `r0`.
#[derive(Clone, Copy, Debug)]
pub struct FlowGeometry {
    pub aspect: f32,
    pub r0: f32,
}

impl FlowGeometry {
    pub fn from_effect(effect: &FlameEffect, baked: &FlameBaked) -> Self {
        Self {
            aspect: branch_aspect(effect),
            r0: branch_trunk_radius_at(effect, baked, 0.0).max(1e-3),
        }
    }
}

struct VortexPair {
    centre_x: f32,
    y: f32,
    circulation: f32,
}

fn burst_velocity(flow: &FlameFlow, plane: usize, time: f32) -> f32 {
    if flow.burst == 0.0 || flow.gust_frequency <= 0.0 {
        return 0.0;
    }
    let period = BURST_PERIODS_PER_GUST / flow.gust_frequency;
    let width = BURST_WIDTH_OVER_PERIOD * period;
    let index = (time / period).floor() as i64;
    let mut velocity = 0.0;
    for i in index - 1..=index + 1 {
        let jitter = hash01(plane as u32 + 7, i, 3) - 0.5;
        let burst_time = (i as f32 + 0.5 + jitter) * period;
        let sign = if hash01(plane as u32 + 7, i, 5) < 0.5 {
            -1.0
        } else {
            1.0
        };
        let phase = (time - burst_time) / width;
        velocity += sign * flow.burst * (-phase * phase).exp();
    }
    velocity
}

/// Lateral gust velocity at the tip for one plane (height-weighted by the caller).
pub fn gust_velocity(flow: &FlameFlow, plane: usize, time: f32) -> f32 {
    let base = flow.gust_frequency.max(0.0) * std::f32::consts::TAU;
    let plane_phase = plane as f32 * 1.7;
    let sway: f32 = GUST_HARMONICS
        .iter()
        .enumerate()
        .map(|(k, (ratio, weight))| {
            weight * (base * ratio * time + plane_phase + k as f32 * 2.1).sin()
        })
        .sum();
    flow.gust * sway / GUST_HARMONICS.iter().map(|(_, w)| w).sum::<f32>()
        + burst_velocity(flow, plane, time)
}

fn active_vortex_pairs(
    flow: &FlameFlow,
    geometry: FlowGeometry,
    plane: usize,
    time: f32,
) -> Vec<VortexPair> {
    if flow.period <= 0.0 || flow.rise <= 0.0 {
        return Vec::new();
    }
    let exit_y = geometry.aspect + 2.0 * flow.core * geometry.r0;
    let rise = flow.rise * geometry.aspect;
    let travel_time = exit_y / rise;
    let first = ((time - travel_time) / flow.period).floor() as i64 - 1;
    let last = (time / flow.period).ceil() as i64 + 1;
    let mut pairs: Vec<VortexPair> = (first..=last)
        .filter_map(|index| {
            let jitter = FLOW_SPAWN_JITTER * (hash01(plane as u32 + 3, index, 11) - 0.5);
            let age = time - (index as f32 + jitter) * flow.period;
            let y = rise * age;
            if age < 0.0 || y >= exit_y {
                return None;
            }
            let sign = if hash01(plane as u32 + 3, index, 13) < 0.5 {
                -1.0
            } else {
                1.0
            };
            let centre_x = 0.3 * geometry.r0 * sign * hash01(plane as u32 + 3, index, 17);
            Some(VortexPair {
                centre_x,
                y,
                circulation: flow.strength * geometry.r0 * geometry.r0,
            })
        })
        .collect();
    pairs.truncate(FLOW_VORTEX_MAX_PAIRS);
    pairs
}

fn vortex_lateral_velocity(
    x: f32,
    y: f32,
    vortex_x: f32,
    vortex_y: f32,
    circulation: f32,
    core: f32,
) -> f32 {
    let dx = x - vortex_x;
    let dy = y - vortex_y;
    let r2 = (dx * dx + dy * dy).max(1e-8);
    let tangential =
        circulation / (std::f32::consts::TAU * r2) * (1.0 - (-r2 / (core * core).max(1e-6)).exp());
    -tangential * dy
}

/// Lateral flow velocity at (x, y): the vortex pairs (left vortex
/// counter-clockwise, right clockwise, so the pair propels itself upward) plus
/// the gust growing with height.
fn inject_weight(flow: &FlameFlow, geometry: FlowGeometry, y: f32) -> f32 {
    if flow.inject_height <= 0.0 {
        y / geometry.aspect.max(1e-3)
    } else {
        (1.0 - y / (flow.inject_height * geometry.aspect).max(1e-3)).clamp(0.0, 1.0)
    }
}

fn lateral_velocity(
    flow: &FlameFlow,
    geometry: FlowGeometry,
    pairs: &[VortexPair],
    x: f32,
    y: f32,
    gust: f32,
) -> f32 {
    let core = (flow.core * geometry.r0).max(1e-3);
    let spacing = geometry.r0;
    let induced: f32 = pairs
        .iter()
        .map(|pair| {
            vortex_lateral_velocity(
                x,
                y,
                pair.centre_x - spacing,
                pair.y,
                pair.circulation,
                core,
            ) + vortex_lateral_velocity(
                x,
                y,
                pair.centre_x + spacing,
                pair.y,
                -pair.circulation,
                core,
            )
        })
        .sum();
    induced + gust * inject_weight(flow, geometry, y)
}

fn transport_carried(
    flow: &FlameFlow,
    geometry: FlowGeometry,
    plane: usize,
    dt: f32,
    src: &[FlowMarker],
    carried: &mut [(f32, f32)],
) {
    let count = src.len();
    for (index, entry) in carried.iter_mut().enumerate() {
        let y = index as f32 / (count - 1) as f32 * geometry.aspect;
        let v = flow.transport_speed * (1.0 + flow.transport_accel * y / geometry.aspect);
        let y_src = y - v * dt;
        entry.0 = if flow.transport_speed == 0.0 {
            src[index].offset[plane]
        } else {
            interpolate_upstream(src, geometry.aspect, y_src, plane)
        };
        if plane == 0 {
            entry.1 = if flow.transport_speed == 0.0 {
                src[index].width_scale
            } else {
                interpolate_upstream_width(src, geometry.aspect, y_src)
            };
        }
    }
}

fn damping_at(flow: &FlameFlow, geometry: FlowGeometry, y: f32) -> f32 {
    flow.damping
        * (1.0
            - flow.damping_slope.clamp(0.0, 1.0) * (y / geometry.aspect.max(1e-3)).clamp(0.0, 1.0))
}

fn injection_write(
    flow: &FlameFlow,
    geometry: FlowGeometry,
    plane: usize,
    time: f32,
    dt: f32,
    carried: &[(f32, f32)],
    dst: &mut [FlowMarker],
) {
    let pairs = active_vortex_pairs(flow, geometry, plane, time);
    let gust = gust_velocity(flow, plane, time);
    for (index, dst_marker) in dst.iter_mut().enumerate() {
        let y = index as f32 / (carried.len() - 1) as f32 * geometry.aspect;
        let x = carried[index].0;
        let carried_width = carried[index].1;
        let half_width = carried_width * geometry.r0;
        let centre = lateral_velocity(flow, geometry, &pairs, x, y, gust);
        let left = lateral_velocity(flow, geometry, &pairs, x - half_width, y, gust);
        let right = lateral_velocity(flow, geometry, &pairs, x + half_width, y, gust);
        let stretch = (right - left) / (2.0 * geometry.r0);
        dst_marker.offset[plane] = x + (centre - damping_at(flow, geometry, y) * x) * dt;
        if plane == 0 {
            dst_marker.width_scale = (carried_width
                + (stretch - damping_at(flow, geometry, y) * (carried_width - 1.0)) * dt)
                .clamp(0.2, 1.6);
        }
    }
}

fn interpolate_upstream(src: &[FlowMarker], aspect: f32, y_src: f32, plane: usize) -> f32 {
    let count = src.len();
    if y_src < 0.0 {
        return 0.0;
    }
    let frac = y_src / aspect * (count - 1) as f32;
    let j = frac.floor() as usize;
    if j >= count - 1 {
        return src[count - 1].offset[plane];
    }
    let t = frac - j as f32;
    src[j].offset[plane] * (1.0 - t) + src[j + 1].offset[plane] * t
}

fn interpolate_upstream_width(src: &[FlowMarker], aspect: f32, y_src: f32) -> f32 {
    let count = src.len();
    if y_src < 0.0 {
        return 1.0;
    }
    let frac = y_src / aspect * (count - 1) as f32;
    let j = frac.floor() as usize;
    if j >= count - 1 {
        return src[count - 1].width_scale;
    }
    let t = frac - j as f32;
    src[j].width_scale * (1.0 - t) + src[j + 1].width_scale * t
}

fn vortex_travel_time(flow: &FlameFlow, geometry: FlowGeometry) -> f32 {
    let exit_y = geometry.aspect + 2.0 * flow.core * geometry.r0;
    exit_y / (flow.rise * geometry.aspect).max(1e-3)
}

/// Integration step and history window follow the flow's own time scale: a
/// step resolves the spawn period, the window outlasts a pair's crossing and
/// the markers' damping memory.
fn simulation_schedule(flow: &FlameFlow, geometry: FlowGeometry) -> (f32, f32) {
    let dt = FLOW_SIM_DT.min(flow.period.max(1e-3) / FLOW_STEPS_PER_PERIOD);
    let memory = FLOW_DAMPING_MEMORY_FOLDS
        / (flow.damping * (1.0 - flow.damping_slope.clamp(0.0, 1.0))).max(1e-3);
    let mut history =
        (2.0 * vortex_travel_time(flow, geometry) + memory).clamp(dt, FLOW_HISTORY_SECONDS);
    if flow.transport_speed > 0.0 {
        let v_min = flow.transport_speed;
        let transport_time = geometry.aspect / v_min;
        history = history.max(transport_time);
    }
    (dt, history)
}

/// Marker column at `time`, derived from (parameters, geometry, time) only:
/// a fixed-step re-simulation over the trailing history window from the rest
/// column, so the result is the same whatever frames were rendered before.
pub fn simulate_flow_markers(
    flow: &FlameFlow,
    geometry: FlowGeometry,
    time: f32,
) -> [FlowMarker; FLOW_MARKER_COUNT] {
    let mut markers = [FlowMarker::default(); FLOW_MARKER_COUNT];
    if flow.gain == 0.0 {
        return markers;
    }
    let (dt, history) = simulation_schedule(flow, geometry);
    let start = (time - history).max(0.0);
    let steps = ((time - start) / dt).ceil().max(0.0) as usize;
    let mut prev: [FlowMarker; FLOW_MARKER_COUNT] = [FlowMarker::default(); FLOW_MARKER_COUNT];
    for step in 0..steps {
        let sim_time = start + step as f32 * dt;
        let mut carried: [(f32, f32); FLOW_MARKER_COUNT] = [(0.0, 1.0); FLOW_MARKER_COUNT];
        for plane in 0..PLANE_COUNT {
            transport_carried(flow, geometry, plane, dt, &prev, &mut carried);
            injection_write(flow, geometry, plane, sim_time, dt, &carried, &mut markers);
        }
        prev.copy_from_slice(&markers);
    }
    markers
}

struct Lobe {
    y: f32,
    side: f32,
    plane: usize,
    amplitude: f32,
    size: f32,
    age: f32,
}

fn lobe_envelope(age01: f32) -> f32 {
    let s = (std::f32::consts::PI * age01.clamp(0.0, 1.0)).sin();
    s * s
}

/// Lobes alive at `time`, derived from (parameters, time) only.
fn lobe_height_amplitude(
    lobe: &FlameLobe,
    flow: &FlameFlow,
    geometry: FlowGeometry,
    spawn_height: f32,
    age: f32,
    life: f32,
) -> (f32, f32) {
    if lobe.transport > 0.0 {
        let y0 = spawn_height.max(0.0);
        let y = if flow.transport_accel == 0.0 {
            y0 + flow.transport_speed * age / geometry.aspect
        } else {
            let drift = 1.0 / flow.transport_accel;
            (y0 + drift)
                * (flow.transport_speed * flow.transport_accel * age / geometry.aspect).exp()
                - drift
        };
        let amplitude = lobe.gain
            * lobe.transport
            * (-damping_at(flow, geometry, y * geometry.aspect) * age).exp();
        (y, amplitude)
    } else {
        let y = if lobe.accel > 0.0 {
            let drift = lobe.rise / lobe.accel;
            (spawn_height.max(0.0) + drift) * (lobe.accel * age).exp() - drift
        } else {
            spawn_height + lobe.rise * age
        };
        let amplitude = lobe.gain * lobe_envelope(age / life);
        (y, amplitude)
    }
}

fn active_lobes(
    lobe: &FlameLobe,
    flow: &FlameFlow,
    geometry: FlowGeometry,
    time: f32,
) -> Vec<Lobe> {
    if lobe.gain == 0.0 || lobe.period <= 0.0 || lobe.life <= 0.0 {
        return Vec::new();
    }
    let life = if lobe.transport > 0.0 {
        FLOW_DAMPING_MEMORY_FOLDS
            / (flow.damping * (1.0 - flow.damping_slope.clamp(0.0, 1.0))).max(1e-3)
    } else {
        lobe.life
    };
    let spread = lobe.spread.clamp(0.0, 1.0);
    let first = ((time - life) / lobe.period).floor() as i64 - 1;
    let last = (time / lobe.period).ceil() as i64 + 1;
    (first..=last)
        .filter_map(|index| {
            let jitter = spread * LOBE_SPAWN_JITTER * (hash01(LOBE_SEED, index, 1) - 0.5);
            let age = time - (index as f32 + jitter) * lobe.period;
            if age < 0.0 || age >= life {
                return None;
            }
            let alternating = if index.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
            let side = if hash01(LOBE_SEED, index, 2) < 0.5 * spread {
                -alternating
            } else {
                alternating
            };
            let plane = if hash01(LOBE_SEED, index, 3) < 0.5 {
                0
            } else {
                1
            };
            let spawn_height = lobe.spawn_height
                + lobe.spawn_range.max(0.0) * hash01(LOBE_SEED, index, 6)
                + spread * LOBE_HEIGHT_SCATTER * (hash01(LOBE_SEED, index, 4) - 0.5);
            let size = lobe.size
                * (1.0 + spread * LOBE_SIZE_SCATTER * (2.0 * hash01(LOBE_SEED, index, 5) - 1.0));

            let (y, amplitude) =
                lobe_height_amplitude(lobe, flow, geometry, spawn_height, age, life);

            if y > 1.0 + size.max(1e-3) {
                return None;
            }

            Some(Lobe {
                y,
                side,
                plane,
                amplitude,
                size: size.max(1e-3),
                age,
            })
        })
        .collect()
}

/// Inject lobes into the marker column using the bump formula.
/// No envelope — uses raw amplitude at birth (the lobe's gain).
fn inject_lobes(
    markers: &mut [FlowMarker; FLOW_MARKER_COUNT],
    lobes: &[Lobe],
    lobe: &FlameLobe,
    geometry: FlowGeometry,
    scale: f32,
) {
    if lobes.is_empty() {
        return;
    }
    let self_shift = lobe.shift.clamp(0.0, 1.0);
    for (index, marker) in markers.iter_mut().enumerate() {
        let h = index as f32 / (FLOW_MARKER_COUNT - 1) as f32;
        for l in lobes {
            let d = (h - l.y) / l.size;
            let bump = l.amplitude * scale * (-d * d).exp();
            marker.offset[l.plane] += l.side * self_shift * bump * geometry.r0;
            marker.width_scale += bump;
        }
    }
}
pub fn build_flow_field(effect: &FlameEffect, baked: &FlameBaked) -> FlameFlowField {
    let geometry = FlowGeometry::from_effect(effect, baked);
    let mut markers = simulate_flow_markers(&effect.flow, geometry, effect.time);
    if effect.flow.gain != 0.0 {
        inject_lobes(
            &mut markers,
            &active_lobes(&effect.lobe, &effect.flow, geometry, effect.time),
            &effect.lobe,
            geometry,
            1.0,
        );
    }
    let mut table = [[0.0f32, 0.0, 1.0, 0.0]; FLOW_MARKER_COUNT];
    for (slot, marker) in table.iter_mut().zip(&markers) {
        *slot = [marker.offset[0], marker.offset[1], marker.width_scale, 0.0];
    }
    FlameFlowField {
        gain: effect.flow.gain,
        count: FLOW_MARKER_COUNT as f32,
        _padding: [0.0; 2],
        markers: table,
    }
}

/// Mirror of flameFlowSample: (offset x, offset z, width scale) at height01.
pub fn flow_sample(field: &FlameFlowField, height01: f32) -> [f32; 3] {
    if field.gain == 0.0 {
        return [0.0, 0.0, 1.0];
    }
    let position = height01.clamp(0.0, 1.0) * (FLOW_MARKER_COUNT - 1) as f32;
    let index = (position.floor() as usize).min(FLOW_MARKER_COUNT - 2);
    let t = position - index as f32;
    let a = field.markers[index];
    let b = field.markers[index + 1];
    [
        field.gain * (a[0] + (b[0] - a[0]) * t),
        field.gain * (a[1] + (b[1] - a[1]) * t),
        1.0 + field.gain * (a[2] + (b[2] - a[2]) * t - 1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flow_on() -> FlameFlow {
        FlameFlow {
            gain: 1.0,
            period: 1.0,
            rise: 0.3,
            strength: 1.0,
            core: 0.6,
            gust: 0.3,
            gust_frequency: 0.4,
            burst: 0.5,
            damping: 0.5,
            damping_slope: 0.0,
            transport_speed: 0.0,
            transport_accel: 0.0,
            inject_height: 0.0,
        }
    }

    fn geometry() -> FlowGeometry {
        FlowGeometry {
            aspect: 2.5,
            r0: 0.7,
        }
    }

    #[test]
    fn gain_zero_is_identity() {
        let mut effect = FlameEffect::default();
        effect.time = 5.0;
        let field = build_flow_field(&effect, &FlameBaked::default());
        assert_eq!(field.gain, 0.0);
        assert_eq!(flow_sample(&field, 0.5), [0.0, 0.0, 1.0]);
        assert!(field.markers.iter().all(|m| *m == [0.0, 0.0, 1.0, 0.0]));
    }

    #[test]
    fn markers_are_a_function_of_time_only() {
        let a = simulate_flow_markers(&flow_on(), geometry(), 7.3);
        let b = simulate_flow_markers(&flow_on(), geometry(), 7.3);
        assert_eq!(a, b);
    }

    #[test]
    fn flow_moves_and_reshapes_the_column() {
        let markers = simulate_flow_markers(&flow_on(), geometry(), 6.0);
        let moved = markers
            .iter()
            .any(|m| m.offset[0].abs() > 1e-3 || m.offset[1].abs() > 1e-3);
        let reshaped = markers.iter().any(|m| (m.width_scale - 1.0).abs() > 1e-3);
        assert!(moved && reshaped);
        assert!(markers
            .iter()
            .all(|m| m.width_scale >= 0.2 && m.offset[0].abs() < 5.0));
    }

    #[test]
    fn lobe_train_is_one_sided_and_off_by_default() {
        let flow = FlameFlow::default();
        let geo = geometry();
        let mut markers = [FlowMarker::default(); FLOW_MARKER_COUNT];
        inject_lobes(
            &mut markers,
            &active_lobes(&FlameLobe::default(), &flow, geo, 3.0),
            &FlameLobe::default(),
            geo,
            1.0,
        );
        let mut markers_defaults_zeroed = [FlowMarker::default(); FLOW_MARKER_COUNT];
        inject_lobes(
            &mut markers_defaults_zeroed,
            &active_lobes(
                &FlameLobe {
                    gain: 1.0,
                    spawn_range: 0.0,
                    accel: 0.0,
                    ..FlameLobe::default()
                },
                &flow,
                geo,
                3.0,
            ),
            &FlameLobe {
                gain: 1.0,
                spawn_range: 0.0,
                accel: 0.0,
                ..FlameLobe::default()
            },
            geo,
            1.0,
        );
        let mut markers_enabled = [FlowMarker::default(); FLOW_MARKER_COUNT];
        inject_lobes(
            &mut markers_enabled,
            &active_lobes(
                &FlameLobe {
                    gain: 1.0,
                    spawn_range: 0.6,
                    accel: 2.0,
                    ..FlameLobe::default()
                },
                &flow,
                geo,
                3.0,
            ),
            &FlameLobe {
                gain: 1.0,
                spawn_range: 0.6,
                accel: 2.0,
                ..FlameLobe::default()
            },
            geo,
            1.0,
        );
        assert_ne!(markers_defaults_zeroed, markers_enabled);
        assert!(markers.iter().all(|m| *m == FlowMarker::default()));

        let lobe = FlameLobe {
            gain: 0.5,
            period: 10.0,
            life: 2.0,
            spread: 0.0,
            ..FlameLobe::default()
        };
        inject_lobes(
            &mut markers,
            &active_lobes(&lobe, &flow, geo, 1.0),
            &lobe,
            geo,
            1.0,
        );
        let peak = markers
            .iter()
            .max_by(|a, b| a.width_scale.total_cmp(&b.width_scale))
            .unwrap();
        assert!(peak.width_scale > 1.05);
        let bulge = (peak.width_scale - 1.0) * geo.r0;
        let offset = peak.offset[0].abs().max(peak.offset[1].abs());
        assert!(
            (offset - bulge).abs() < 1e-5,
            "offset {offset} bulge {bulge}"
        );
    }

    #[test]
    fn base_marker_stays_put() {
        let markers = simulate_flow_markers(&flow_on(), geometry(), 6.0);
        assert!(markers[0].offset[0].abs() < 0.2 && markers[0].offset[1].abs() < 0.2);
    }

    #[test]
    fn identity_with_old_impl_at_zero_transport_speed() {
        let flow = flow_on();
        let geo = geometry();
        let time = 6.0;
        let (dt, history) = simulation_schedule(&flow, geo);
        let start = (time - history).max(0.0);
        let steps = ((time - start) / dt).ceil().max(0.0) as usize;

        // Reference: old in-place loop with offset += (centre - damping * x) * dt
        let mut ref_markers: [FlowMarker; FLOW_MARKER_COUNT] =
            [FlowMarker::default(); FLOW_MARKER_COUNT];
        for step in 0..steps {
            let sim_time = start + step as f32 * dt;
            for plane in 0..PLANE_COUNT {
                let pairs = active_vortex_pairs(&flow, geo, plane, sim_time);
                let gust = gust_velocity(&flow, plane, sim_time);
                let count = ref_markers.len();
                for (index, marker) in ref_markers.iter_mut().enumerate() {
                    let y = index as f32 / (count - 1) as f32 * geo.aspect;
                    let x = marker.offset[plane];
                    let half_width = marker.width_scale * geo.r0;
                    let centre = lateral_velocity(&flow, geo, &pairs, x, y, gust);
                    let left = lateral_velocity(&flow, geo, &pairs, x - half_width, y, gust);
                    let right = lateral_velocity(&flow, geo, &pairs, x + half_width, y, gust);
                    let stretch = (right - left) / (2.0 * geo.r0);
                    marker.offset[plane] += (centre - flow.damping * x) * dt;
                    if plane == 0 {
                        marker.width_scale = (marker.width_scale
                            + (stretch - flow.damping * (marker.width_scale - 1.0)) * dt)
                            .clamp(0.2, 1.6);
                    }
                }
            }
        }

        let new_markers = simulate_flow_markers(&flow, geo, time);
        for i in 0..FLOW_MARKER_COUNT {
            assert!(
                (new_markers[i].offset[0] - ref_markers[i].offset[0]).abs() < 1e-6,
                "marker[{i}] offset[0]: new={:?} ref={:?}",
                new_markers[i].offset[0],
                ref_markers[i].offset[0]
            );
            assert!(
                (new_markers[i].offset[1] - ref_markers[i].offset[1]).abs() < 1e-6,
                "marker[{i}] offset[1]: new={:?} ref={:?}",
                new_markers[i].offset[1],
                ref_markers[i].offset[1]
            );
            assert!(
                (new_markers[i].width_scale - ref_markers[i].width_scale).abs() < 1e-6,
                "marker[{i}] width_scale: new={:?} ref={:?}",
                new_markers[i].width_scale,
                ref_markers[i].width_scale
            );
        }
    }

    #[test]
    fn inject_weight_zero_matches_legacy() {
        let flow = FlameFlow {
            inject_height: 0.0,
            ..FlameFlow::default()
        };
        let geo = FlowGeometry {
            aspect: 2.5,
            r0: 0.7,
        };
        for i in 0..=10 {
            let y = i as f32 * 0.25;
            let weight = inject_weight(&flow, geo, y);
            let legacy = y / geo.aspect.max(1e-3);
            assert!(
                (weight - legacy).abs() < 1e-9,
                "y={}: inject_weight={:.6} legacy={:.6}",
                y,
                weight,
                legacy
            );
        }
    }

    #[test]
    fn inject_weight_root_focused() {
        let aspect = 2.5;
        let flow = FlameFlow {
            inject_height: 0.25,
            ..FlameFlow::default()
        };
        let geo = FlowGeometry { aspect, r0: 0.7 };

        // At y=0, weight should be 1.0
        let w0 = inject_weight(&flow, geo, 0.0);
        assert!((w0 - 1.0).abs() < 1e-9, "y=0: expected 1.0, got {:.6}", w0);

        // At y >= 0.25 * aspect, weight should be 0.0
        let boundary_y = 0.25 * aspect;
        let w_boundary = inject_weight(&flow, geo, boundary_y);
        assert!(
            (w_boundary - 0.0).abs() < 1e-9,
            "y={}: expected 0.0, got {:.6}",
            boundary_y,
            w_boundary
        );

        // At y > boundary, weight should also be 0.0
        let w_past = inject_weight(&flow, geo, boundary_y + 1.0);
        assert!(
            (w_past - 0.0).abs() < 1e-9,
            "y={}: expected 0.0, got {:.6}",
            boundary_y + 1.0,
            w_past
        );

        // At y = 0.125 * aspect (halfway), weight should be 0.5
        let mid_y = 0.125 * aspect;
        let w_mid = inject_weight(&flow, geo, mid_y);
        assert!(
            (w_mid - 0.5).abs() < 1e-9,
            "y={}: expected 0.5, got {:.6}",
            mid_y,
            w_mid
        );
    }

    #[test]
    fn lobe_transport_moves_and_damps() {
        let flow = FlameFlow {
            gain: 1.0,
            gust: 0.0,
            burst: 0.0,
            strength: 0.0,
            damping: 0.5,
            transport_speed: 1.0,
            inject_height: 0.0,
            ..FlameFlow::default()
        };
        let geo = geometry();
        let lobe = FlameLobe {
            gain: 1.0,
            period: 100.0,
            life: 2.0,
            rise: 0.0,
            size: 0.08,
            spawn_height: 0.2,
            spawn_range: 0.0,
            accel: 0.0,
            spread: 0.0,
            shift: 1.0,
            transport: 1.0,
        };

        let lobes_early = active_lobes(&lobe, &flow, geo, 0.5);
        assert_eq!(
            lobes_early.len(),
            1,
            "expected exactly one lobe at time 0.5"
        );
        let lobe_early = &lobes_early[0];

        let lobes_later = active_lobes(&lobe, &flow, geo, 1.5);
        assert_eq!(
            lobes_later.len(),
            1,
            "expected exactly one lobe at time 1.5"
        );
        let lobe_later = &lobes_later[0];

        assert!(
            lobe_later.y > lobe_early.y,
            "y should increase: early={:.4} later={:.4}",
            lobe_early.y,
            lobe_later.y
        );

        let ratio = lobe_later.amplitude / lobe_early.amplitude;
        let expected_ratio = (-0.5_f32).exp();
        assert!(
            (ratio - expected_ratio).abs() < 1e-4,
            "amplitude ratio {:.6} should match exp(-0.5)={:.6}",
            ratio,
            expected_ratio
        );
    }

    #[test]
    fn lobe_transport_beyond_lobe_life() {
        let geo = geometry();
        let flow = FlameFlow {
            gain: 1.0,
            gust: 0.0,
            burst: 0.0,
            strength: 0.0,
            damping: 0.5,
            transport_speed: 1.0,
            inject_height: 0.0,
            ..FlameFlow::default()
        };
        let lobe = FlameLobe {
            gain: 1.0,
            period: 100.0,
            life: 0.048,
            rise: 0.0,
            size: 0.08,
            spawn_height: 0.2,
            spawn_range: 0.0,
            accel: 0.0,
            spread: 0.0,
            shift: 1.0,
            transport: 1.0,
        };

        let lobes = active_lobes(&lobe, &flow, geo, 1.0);
        assert_eq!(
            lobes.len(),
            1,
            "expected exactly one lobe at time 1.0 (age > lobe.life, but effective life comes from damping)"
       );
    }

    #[test]
    fn damping_at_slope_zero_is_uniform() {
        let flow = FlameFlow {
            damping: 0.5,
            damping_slope: 0.0,
            ..FlameFlow::default()
        };
        let geo = FlowGeometry {
            aspect: 2.5,
            r0: 1.0,
        };
        for &y in &[0.0, 1.0, 2.5, 5.0] {
            let d = damping_at(&flow, geo, y);
            assert!(
                (d - flow.damping).abs() < 1e-6,
                "slope 0: expected damping {} at y={}, got {}",
                flow.damping,
                y,
                d
            );
        }
    }

    #[test]
    fn damping_at_slope_half_halves_at_tip() {
        let flow = FlameFlow {
            damping: 0.5,
            damping_slope: 0.5,
            ..FlameFlow::default()
        };
        let geo = FlowGeometry {
            aspect: 2.5,
            r0: 1.0,
        };
        assert!(
            (damping_at(&flow, geo, 0.0) - flow.damping).abs() < 1e-6,
            "y=0: expected {}, got {}",
            flow.damping,
            damping_at(&flow, geo, 0.0)
        );
        assert!(
            (damping_at(&flow, geo, geo.aspect) - flow.damping / 2.0).abs() < 1e-6,
            "y=aspect: expected {}, got {}",
            flow.damping / 2.0,
            damping_at(&flow, geo, geo.aspect)
        );
        assert!(
            (damping_at(&flow, geo, 2.0 * geo.aspect) - flow.damping / 2.0).abs() < 1e-6,
            "y=2*aspect: expected {}, got {}",
            flow.damping / 2.0,
            damping_at(&flow, geo, 2.0 * geo.aspect)
        );
    }
}
