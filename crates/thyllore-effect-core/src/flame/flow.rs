use super::*;
use crate::flame::branch::hash01;

const GUST_HARMONICS: [(f32, f32); 3] = [(1.0, 1.0), (1.618, 0.5), (2.414, 0.3)];
const BURST_PERIODS_PER_GUST: f32 = 10.0;
const BURST_WIDTH_OVER_PERIOD: f32 = 0.08;
const PLANE_COUNT: usize = 2;
const FLOW_STEPS_PER_PERIOD: f32 = 40.0;
const FLOW_DAMPING_MEMORY_FOLDS: f32 = 5.0;

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
    induced + gust * (y / geometry.aspect.max(1e-3))
}

fn step_plane(
    flow: &FlameFlow,
    geometry: FlowGeometry,
    plane: usize,
    time: f32,
    dt: f32,
    markers: &mut [FlowMarker],
) {
    let pairs = active_vortex_pairs(flow, geometry, plane, time);
    let gust = gust_velocity(flow, plane, time);
    let count = markers.len();
    for (index, marker) in markers.iter_mut().enumerate() {
        let y = index as f32 / (count - 1) as f32 * geometry.aspect;
        let x = marker.offset[plane];
        let half_width = marker.width_scale * geometry.r0;
        let centre = lateral_velocity(flow, geometry, &pairs, x, y, gust);
        let left = lateral_velocity(flow, geometry, &pairs, x - half_width, y, gust);
        let right = lateral_velocity(flow, geometry, &pairs, x + half_width, y, gust);
        let stretch = (right - left) / (2.0 * geometry.r0);
        marker.offset[plane] += (centre - flow.damping * x) * dt;
        if plane == 0 {
            marker.width_scale = (marker.width_scale
                + (stretch - flow.damping * (marker.width_scale - 1.0)) * dt)
                .clamp(0.2, 1.6);
        }
    }
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
    let memory = FLOW_DAMPING_MEMORY_FOLDS / flow.damping.max(1e-3);
    let history =
        (2.0 * vortex_travel_time(flow, geometry) + memory).clamp(dt, FLOW_HISTORY_SECONDS);
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
    for step in 0..steps {
        let sim_time = start + step as f32 * dt;
        for plane in 0..PLANE_COUNT {
            step_plane(flow, geometry, plane, sim_time, dt, &mut markers);
        }
    }
    markers
}

pub fn build_flow_field(effect: &FlameEffect, baked: &FlameBaked) -> FlameFlowField {
    let geometry = FlowGeometry::from_effect(effect, baked);
    let markers = simulate_flow_markers(&effect.flow, geometry, effect.time);
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
    fn base_marker_stays_put() {
        let markers = simulate_flow_markers(&flow_on(), geometry(), 6.0);
        assert!(markers[0].offset[0].abs() < 0.2 && markers[0].offset[1].abs() < 0.2);
    }
}
