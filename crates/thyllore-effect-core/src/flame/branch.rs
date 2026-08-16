use super::*;
use crate::flame_radial::{
    flame_radial_radius_scale, flame_radial_support_radius, FlameRadialTaper,
};
use std::f32::consts::TAU;
use thyllore_math_core::dot3;

/// Proxy widening (radial and above the top) that keeps transported density inside
/// the shell cone, in flame-local units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlameProxyPad {
    pub radial: f32,
    pub top: f32,
}

/// One vortex element resolved at a time: a vortex line through `center` along
/// `line`, rotating the plane spanned by `outward` and `up`. The frame is
/// orthonormal in the isotropic coordinates (local y scaled by `aspect`);
/// `line` is tilted out of the horizontal by the element's tilt lane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VortexElement {
    pub center: [f32; 3],
    pub outward: [f32; 3],
    pub line: [f32; 3],
    pub up: [f32; 3],
    pub reach: f32,
    pub core_radius: f32,
    pub circulation: f32,
    pub aspect: f32,
    /// Window center along the line, in reach units.
    pub along_offset: f32,
}

/// Isotropic offset of `p` from the element center (y scaled by aspect).
fn vortex_isotropic_offset(element: &VortexElement, p: [f32; 3]) -> [f32; 3] {
    [
        p[0] - element.center[0],
        (p[1] - element.center[1]) * element.aspect,
        p[2] - element.center[2],
    ]
}

/// (u, along, v) frame coordinates of an isotropic offset.
fn vortex_frame_coordinates(element: &VortexElement, q: [f32; 3]) -> (f32, f32, f32) {
    (
        dot3(q, element.outward),
        dot3(q, element.line) - element.along_offset * element.reach,
        dot3(q, element.up),
    )
}

fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn hash01(seed: u32, index: i64, lane: u32) -> f32 {
    let mixed =
        hash_u32(seed ^ hash_u32((index as u32) ^ hash_u32(lane.wrapping_add(0x9e37_79b9))));
    (mixed >> 8) as f32 / (1u32 << 24) as f32
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Spawn period raised so at most `BRANCH_MAX_ELEMENTS` elements are ever alive:
/// a shorter period saturates the table instead of shortening every element's
/// life, so the tongues keep their size and duration.
pub fn effective_branch_period(branch: &FlameBranch) -> f32 {
    if branch.period <= 0.0 {
        return branch.period;
    }
    branch
        .period
        .max(branch.life.max(0.0) / (BRANCH_MAX_ELEMENTS as f32 - 1.0))
}

fn spawn_branch_element(
    branch: &FlameBranch,
    index: i64,
    trunk_radius_at: &dyn Fn(f32) -> f32,
) -> FlameBranchElement {
    let spread = branch.spread.clamp(0.0, 1.0);
    let seed = branch.seed;
    let period = effective_branch_period(branch);
    let jitter = spread * BRANCH_JITTER_RANGE * (hash01(seed, index, 0) - 0.5);
    let alternating = if index.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
    let side = if hash01(seed, index, 1) < 0.5 * spread {
        -alternating
    } else {
        alternating
    };
    let spawn_height = branch.spawn_height + branch.spawn_range * (hash01(seed, index, 3) - 0.5);
    let scatter = |lane: u32, range: f32| spread * range * (2.0 * hash01(seed, index, lane) - 1.0);
    FlameBranchElement {
        spawn_time: (index as f32 + jitter) * period,
        side,
        azimuth: branch_element_azimuth(index, spread, hash01(seed, index, 2)),
        spawn_height,
        size: 1.0 + scatter(5, BRANCH_SIZE_SCATTER),
        tilt: scatter(6, BRANCH_TILT_RANGE),
        along_offset: scatter(7, BRANCH_ALONG_OFFSET_RANGE),
        hash01: hash01(seed, index, 4),
        trunk_radius: trunk_radius_at(spawn_height.clamp(0.0, 1.0)),
        _padding: [0.0; 3],
    }
}

/// Golden-angle sequence around the full circle, so consecutive tongues leave the
/// trunk in well-separated directions and the column reads the same from every
/// azimuth; `spread` only jitters each element about its slot.
fn branch_element_azimuth(index: i64, spread: f32, jitter01: f32) -> f32 {
    let slot =
        (index as f64 * BRANCH_AZIMUTH_GOLDEN_ANGLE).rem_euclid(std::f64::consts::TAU) as f32;
    slot + spread * BRANCH_AZIMUTH_JITTER * (jitter01 - 0.5)
}

/// Trunk support radius at a normalized height, in flame-local units.
pub fn branch_trunk_radius_at(effect: &FlameEffect, baked: &FlameBaked, height01: f32) -> f32 {
    flame_radial_support_radius(effect.radial_sharpness, effect.support_margin)
        * flame_radial_radius_scale(height01, FlameRadialTaper::from_effect(effect, baked))
}

fn branch_trunk_radius_max(effect: &FlameEffect, baked: &FlameBaked) -> f32 {
    (0..=16)
        .map(|i| branch_trunk_radius_at(effect, baked, i as f32 / 16.0))
        .fold(0.0, f32::max)
}

/// Elements alive at `time`, newest first; derived from (parameters, time) only.
pub fn active_branch_elements(
    branch: &FlameBranch,
    time: f32,
    trunk_radius_at: &dyn Fn(f32) -> f32,
) -> Vec<FlameBranchElement> {
    if branch.period <= 0.0 || branch.gain == 0.0 {
        return Vec::new();
    }
    let life = branch.life.max(0.0);
    if life <= 0.0 {
        return Vec::new();
    }

    let period = effective_branch_period(branch);
    let first = ((time - life) / period).floor() as i64 - 1;
    let last = (time / period).ceil() as i64 + 1;
    let mut elements: Vec<FlameBranchElement> = (first..=last)
        .map(|index| spawn_branch_element(branch, index, trunk_radius_at))
        .filter(|element| {
            let age = time - element.spawn_time;
            age >= 0.0 && age < life
        })
        .collect();
    elements.sort_by(|a, b| b.spawn_time.total_cmp(&a.spawn_time));
    elements.truncate(BRANCH_MAX_ELEMENTS);
    elements
}

/// Rise rate of the visible noise pattern in local height units per second: the
/// advect chain moves the pattern by rise_speed / (aniso_y * noise_frequency).
pub fn branch_rise_rate(effect: &FlameEffect) -> f32 {
    let pattern_scale = effective_noise_aniso_y(&effect.noise, effect.height, effect.radius)
        * effect.noise.frequency;
    effect.warp.rise_speed / pattern_scale.max(1e-3)
}

/// Circulation that turns the core by `gain` radians: gain is the peak rotation
/// angle, independent of the core radius.
pub fn branch_circulation(gain: f32, core_radius: f32) -> f32 {
    gain * TAU * core_radius * core_radius
}

/// Winding envelope: the core angle eases out over the first
/// `BRANCH_WIND_FRACTION` of the life (fastest at birth, decelerating to rest),
/// holds, then unwinds over `envelope_time` so the map is the identity at death.
/// The unwind is hidden outside the trunk by the burnout mask, so only the trunk
/// gap is seen healing.
pub fn branch_envelope(age: f32, life: f32, envelope_time: f32) -> f32 {
    let winding_time = (BRANCH_WIND_FRACTION * life).max(1e-3);
    let t = (age / winding_time).clamp(0.0, 1.0);
    let ease_out = 1.0 - (1.0 - t) * (1.0 - t);
    ease_out * (1.0 - smoothstep(life - envelope_time, life, age))
}

/// Burnout strength: rises from `BRANCH_BURNOUT_START_FRACTION` of the life to 1
/// when the unwind starts, and releases in the last part of the unwind, when the
/// remaining rotation is negligible, so the mask never jumps at death.
pub fn branch_burnout(age: f32, life: f32, envelope_time: f32) -> f32 {
    let unwind_start = life - envelope_time;
    let release_start = life - BRANCH_BURNOUT_RELEASE_FRACTION * envelope_time;
    smoothstep(BRANCH_BURNOUT_START_FRACTION * life, unwind_start, age)
        * (1.0 - smoothstep(release_start, life, age))
}

/// Density mask of one element at trunk-local `p` (before the pull-back): a
/// plateau over the element's disc that only bites the medium outside the
/// trunk (`r > BRANCH_BURNOUT_TRUNK_INNER * S`), so the tongue dims away in place
/// while the trunk keeps its material.
pub fn vortex_burnout_mask(
    element: &VortexElement,
    burnout: f32,
    trunk_radius: f32,
    p: [f32; 3],
) -> f32 {
    let (u, along, v) = vortex_frame_coordinates(element, vortex_isotropic_offset(element, p));
    let reach = element.reach.max(1e-4);
    let outer = 1.0 + BRANCH_BURNOUT_MARGIN;
    let radius = (u * u + v * v + along * along).sqrt() / reach;
    let plateau = 1.0 - smoothstep(1.0, outer, radius);

    let axis_radius = (p[0] * p[0] + p[2] * p[2]).sqrt() / trunk_radius.max(1e-4);
    let outside_trunk = smoothstep(BRANCH_BURNOUT_TRUNK_INNER, 1.0, axis_radius);
    1.0 - burnout * plateau * outside_trunk
}

/// Product of the burnout masks of every live element at trunk-local `p`.
pub fn branch_burnout_mask(field: &FlameBranchField, p: [f32; 3], time: f32) -> f32 {
    let count = (field.count as usize).min(BRANCH_MAX_ELEMENTS);
    field.elements[..count]
        .iter()
        .filter_map(|element| {
            vortex_element_at(field, element, time).map(|vortex| (element, vortex))
        })
        .fold(1.0, |mask, (element, vortex)| {
            let age = time - element.spawn_time;
            let burnout = branch_burnout(age, field.life, field.envelope_time);
            mask * vortex_burnout_mask(&vortex, burnout, element.trunk_radius, p)
        })
}

/// Lamb-Oseen angular displacement per unit circulation and its derivative in
/// rho^2: (1 - exp(-rho^2 / rc^2)) / (2 pi rho^2), finite at the core.
fn lamb_oseen(rho_sq: f32, core_radius: f32) -> (f32, f32) {
    let core_sq = core_radius * core_radius;
    let x = rho_sq / core_sq;
    if x < 1e-3 {
        return (
            (1.0 - 0.5 * x + x * x / 6.0) / (TAU * core_sq),
            (-0.5 + x / 3.0) / (TAU * core_sq * core_sq),
        );
    }
    let decay = (-x).exp();
    let value = (1.0 - decay) / (TAU * rho_sq);
    let derivative = (decay * x - (1.0 - decay)) / (TAU * rho_sq * rho_sq);
    (value, derivative)
}

pub fn vortex_element_at(
    field: &FlameBranchField,
    element: &FlameBranchElement,
    time: f32,
) -> Option<VortexElement> {
    let age = time - element.spawn_time;
    if age < 0.0 || age >= field.life {
        return None;
    }
    let (sin_az, cos_az) = element.azimuth.sin_cos();
    let lateral =
        element.side * element.trunk_radius * (field.core_offset + field.drift_rate * age);
    let center = [
        lateral * cos_az,
        element.spawn_height + field.rise_rate * age,
        lateral * sin_az,
    ];
    let (sin_tilt, cos_tilt) = element.tilt.sin_cos();
    let horizontal_line = [-sin_az, 0.0, cos_az];
    let line = [
        cos_tilt * horizontal_line[0],
        sin_tilt,
        cos_tilt * horizontal_line[2],
    ];
    let up = [
        -sin_tilt * horizontal_line[0],
        cos_tilt,
        -sin_tilt * horizontal_line[2],
    ];

    let progress = age / field.life;
    let reach_ratio = field.reach_start + (field.reach_end - field.reach_start) * progress;
    let scale = element.trunk_radius * element.size;
    let core_radius = field.core_radius * scale;
    Some(VortexElement {
        center,
        outward: [cos_az, 0.0, sin_az],
        line,
        up,
        reach: reach_ratio * scale,
        core_radius,
        circulation: element.side
            * branch_circulation(field.gain, core_radius)
            * branch_envelope(age, field.life, field.envelope_time),
        aspect: field.aspect,
        along_offset: element.along_offset,
    })
}

/// Pull-back through one element with its Jacobian-vector product along `dir`:
/// each slice perpendicular to the vortex line rotates about the line by the
/// Lamb-Oseen angle gated by a ball `rho^2 + along^2 < reach^2` around the
/// element center (a per-slice rotation, so the map is a bijection with unit
/// determinant; the ball keeps the tongue's boundary round from every view).
pub fn vortex_pull_back_jvp(
    element: &VortexElement,
    p: [f32; 3],
    dir: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let aspect = element.aspect;
    let q = vortex_isotropic_offset(element, p);
    let (u, along, v) = vortex_frame_coordinates(element, q);
    let reach = element.reach;
    let reach_sq = reach * reach;
    let rho_sq = u * u + v * v;
    let s = (rho_sq + along * along) / reach_sq;
    if s >= 1.0 {
        return (p, dir);
    }

    let gate = (1.0 - s) * (1.0 - s);
    let (profile, d_profile) = lamb_oseen(rho_sq, element.core_radius);
    let circulation = element.circulation;
    let psi = circulation * gate * profile;

    let dq = [dir[0], dir[1] * aspect, dir[2]];
    let du = dot3(dq, element.outward);
    let d_along = dot3(dq, element.line);
    let dv = dot3(dq, element.up);
    let d_rho_sq = 2.0 * (u * du + v * dv);
    let d_s = (d_rho_sq + 2.0 * along * d_along) / reach_sq;
    let d_gate = -2.0 * (1.0 - s) * d_s;
    let d_psi = circulation * (d_gate * profile + gate * d_profile * d_rho_sq);

    let (sn, cs) = psi.sin_cos();
    let u1 = u * cs - v * sn;
    let v1 = u * sn + v * cs;
    let du1 = du * cs - dv * sn - d_psi * v1;
    let dv1 = du * sn + dv * cs + d_psi * u1;
    let along_total = along + element.along_offset * reach;
    let [ex, ey, ez] = element.outward;
    let [lx, ly, lz] = element.line;
    let [vx, vy, vz] = element.up;
    (
        [
            element.center[0] + u1 * ex + along_total * lx + v1 * vx,
            element.center[1] + (u1 * ey + along_total * ly + v1 * vy) / aspect,
            element.center[2] + u1 * ez + along_total * lz + v1 * vz,
        ],
        [
            du1 * ex + d_along * lx + dv1 * vx,
            (du1 * ey + d_along * ly + dv1 * vy) / aspect,
            du1 * ez + d_along * lz + dv1 * vz,
        ],
    )
}

pub fn vortex_pull_back(element: &VortexElement, p: [f32; 3]) -> [f32; 3] {
    vortex_pull_back_jvp(element, p, [0.0; 3]).0
}

/// Composite pull-back through every live element, newest first, with the JVP.
pub fn branch_pull_back_jvp(
    field: &FlameBranchField,
    p: [f32; 3],
    dir: [f32; 3],
    time: f32,
) -> ([f32; 3], [f32; 3]) {
    let count = (field.count as usize).min(BRANCH_MAX_ELEMENTS);
    field.elements[..count]
        .iter()
        .filter_map(|element| vortex_element_at(field, element, time))
        .fold((p, dir), |(p, dir), element| {
            vortex_pull_back_jvp(&element, p, dir)
        })
}

pub fn branch_pull_back(field: &FlameBranchField, p: [f32; 3], time: f32) -> [f32; 3] {
    branch_pull_back_jvp(field, p, [0.0; 3], time).0
}

/// Radius of the ball one element can move medium within, at its largest
/// (end of life, largest size lane), in flame-local units.
fn branch_ball_radius_max(branch: &FlameBranch, trunk_radius: f32) -> f32 {
    let size_max = 1.0 + branch.spread.clamp(0.0, 1.0) * BRANCH_SIZE_SCATTER;
    branch.reach.max(1e-3) * size_max * trunk_radius
}

/// Radial proxy pad the shader cone actually uses (`clampToShellCone` radiusPad):
/// the branch pad plus the meander sway, which displaces the whole column by up
/// to `meander_amp` at the tip in either direction. The scissor and picking must
/// widen by the same amount, or steep views clip the swaying column.
pub fn flame_proxy_radial_pad(branch_pad_radial: f32, meander_amp: f32) -> f32 {
    branch_pad_radial + 2.0 * meander_amp.abs()
}

/// Full proxy pad of an effect: branch element geometry plus the meander sway.
pub fn flame_proxy_pad(effect: &FlameEffect, baked: &FlameBaked) -> FlameProxyPad {
    let branch = branch_proxy_pad(effect, baked);
    FlameProxyPad {
        radial: flame_proxy_radial_pad(branch.radial, effect.meander.amp),
        top: branch.top,
    }
}

/// Proxy pad from the element geometry alone: transported density only appears
/// inside an element's ball, so the widest lateral core position plus the ball
/// radius bounds it sideways, and the highest core position plus the ball's
/// height bounds it above the top.
pub fn branch_proxy_pad(effect: &FlameEffect, baked: &FlameBaked) -> FlameProxyPad {
    let branch = &effect.branch;
    if branch.period <= 0.0 || branch.gain == 0.0 {
        return FlameProxyPad::default();
    }
    let trunk_radius = branch_trunk_radius_max(effect, baked);
    let ball = branch_ball_radius_max(branch, trunk_radius);
    let lateral = (branch.core_offset.abs() + BRANCH_DRIFT_OVER_LIFE) * trunk_radius;
    let center_top = branch.spawn_height
        + 0.5 * branch.spawn_range.abs()
        + branch_rise_rate(effect) * branch.life.max(0.0);
    FlameProxyPad {
        radial: lateral + ball,
        top: (center_top + ball / branch_aspect(effect) - 1.0).max(0.0),
    }
}

/// Height over bounding radius: the transport is isotropic in world units, so
/// local y is scaled by this before the meridional rotation.
pub fn branch_aspect(effect: &FlameEffect) -> f32 {
    effect.height.max(MIN_FLAME_EXTENT) / flame_bounding_radius(effect).max(MIN_FLAME_EXTENT)
}

pub fn build_branch_field(effect: &FlameEffect, baked: &FlameBaked) -> FlameBranchField {
    let branch = &effect.branch;
    let life = branch.life.max(0.0);
    let mut elements = [FlameBranchElement::default(); BRANCH_MAX_ELEMENTS];
    let trunk_radius_at = |height01: f32| branch_trunk_radius_at(effect, baked, height01);
    let active = active_branch_elements(branch, effect.time, &trunk_radius_at);
    elements[..active.len()].copy_from_slice(&active);
    let pad = branch_proxy_pad(effect, baked);
    FlameBranchField {
        count: active.len() as f32,
        period: effective_branch_period(branch),
        life,
        gain: branch.gain,
        rise_rate: branch_rise_rate(effect),
        drift_rate: BRANCH_DRIFT_OVER_LIFE / life.max(1e-3),
        aspect: branch_aspect(effect),
        core_radius: branch.core_radius.max(1e-3),
        reach_start: BRANCH_REACH_GROWTH_START * branch.reach.max(1e-3),
        reach_end: branch.reach.max(1e-3),
        envelope_time: BRANCH_ENVELOPE_FRACTION * life,
        core_offset: branch.core_offset,
        bounding_pad: pad.radial,
        bounding_pad_y: pad.top,
        _padding1: [0.0; 2],
        age_profile: FlameBranchAgeProfile::default(),
        elements,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_trunk(_height01: f32) -> f32 {
        1.0
    }

    fn branch_on() -> FlameBranch {
        FlameBranch {
            period: 0.4,
            life: 2.5,
            gain: 1.5,
            core_radius: 0.35,
            core_offset: 0.0,
            reach: 1.5,
            spread: 0.3,
            spawn_height: 0.35,
            spawn_range: 0.4,
            seed: 7,
        }
    }

    fn effect_with_branches() -> FlameEffect {
        let mut effect = FlameEffect::default();
        effect.branch = branch_on();
        effect.time = 3.7;
        effect
    }

    fn frame_element(center: [f32; 3], azimuth: f32, tilt: f32, aspect: f32) -> VortexElement {
        let (sin_az, cos_az) = azimuth.sin_cos();
        let (sin_tilt, cos_tilt) = tilt.sin_cos();
        let horizontal = [-sin_az, 0.0, cos_az];
        VortexElement {
            center,
            outward: [cos_az, 0.0, sin_az],
            line: [cos_tilt * horizontal[0], sin_tilt, cos_tilt * horizontal[2]],
            up: [
                -sin_tilt * horizontal[0],
                cos_tilt,
                -sin_tilt * horizontal[2],
            ],
            reach: 0.9,
            core_radius: 0.35,
            circulation: 2.0,
            aspect,
            along_offset: 0.0,
        }
    }

    fn sample_element() -> VortexElement {
        let mut element = frame_element([0.1, 0.5, -0.05], 0.3, 0.35, 2.5);
        element.along_offset = 0.2;
        element
    }

    fn sample_points() -> Vec<[f32; 3]> {
        let mut points = Vec::new();
        for i in 0..7 {
            for j in 0..7 {
                for k in 0..5 {
                    points.push([
                        -1.5 + 0.5 * i as f32,
                        0.1 + 0.15 * k as f32,
                        -1.5 + 0.5 * j as f32,
                    ]);
                }
            }
        }
        points
    }

    fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    #[test]
    fn test_no_elements_when_period_or_gain_is_zero() {
        let mut branch = branch_on();
        branch.period = 0.0;
        assert!(active_branch_elements(&branch, 5.0, &unit_trunk).is_empty());
        let mut branch = branch_on();
        branch.gain = 0.0;
        assert!(active_branch_elements(&branch, 5.0, &unit_trunk).is_empty());
        let mut effect = FlameEffect::default();
        effect.time = 5.0;
        assert_eq!(
            build_branch_field(&effect, &FlameBaked::default()).count,
            0.0
        );
        assert_eq!(
            branch_proxy_pad(&effect, &FlameBaked::default()),
            FlameProxyPad::default()
        );
    }

    #[test]
    fn test_element_table_is_deterministic_and_bounded() {
        let branch = branch_on();
        for time in [0.0_f32, 1.3, 7.77, 123.4] {
            let once = active_branch_elements(&branch, time, &unit_trunk);
            let twice = active_branch_elements(&branch, time, &unit_trunk);
            assert_eq!(once, twice);
            assert!(!once.is_empty());
            assert!(once.len() <= BRANCH_MAX_ELEMENTS);
            for pair in once.windows(2) {
                assert!(pair[0].spawn_time > pair[1].spawn_time, "newest first");
            }
            for element in &once {
                let age = time - element.spawn_time;
                assert!(age >= 0.0 && age < branch.life);
                assert!(element.side == 1.0 || element.side == -1.0);
            }
        }
    }

    #[test]
    fn test_period_clamp_keeps_table_within_capacity_without_dropping() {
        let mut branch = branch_on();
        branch.period = 0.05;
        branch.life = 10.0;
        branch.spread = 1.0;
        let life = branch.life;
        let period = effective_branch_period(&branch);
        assert!((period - 10.0 / (BRANCH_MAX_ELEMENTS as f32 - 1.0)).abs() < 1e-6);
        let mut step = 0;
        while step < 400 {
            let time = step as f32 * 0.013;
            let elements = active_branch_elements(&branch, time, &unit_trunk);
            assert!(elements.len() <= BRANCH_MAX_ELEMENTS);
            let alive = ((time - life) / period).floor() as i64..=(time / period).ceil() as i64;
            let unclipped = alive
                .map(|index| spawn_branch_element(&branch, index, &unit_trunk))
                .filter(|e| time - e.spawn_time >= 0.0 && time - e.spawn_time < life)
                .count();
            assert_eq!(
                elements.len(),
                unclipped,
                "an element was truncated at t={time}"
            );
            step += 1;
        }
    }

    #[test]
    fn test_pull_back_is_identity_outside_the_disc_and_at_zero_circulation() {
        let element = sample_element();
        let far = [3.0, 0.5, 0.0];
        assert_eq!(vortex_pull_back(&element, far), far);
        let mut off = element;
        off.circulation = 0.0;
        for p in sample_points() {
            assert!(distance(vortex_pull_back(&off, p), p) < 1e-6);
        }
    }

    #[test]
    fn test_pull_back_round_trips_through_the_forward_map() {
        let element = sample_element();
        let mut forward = element;
        forward.circulation = -element.circulation;
        for p in sample_points() {
            let pulled = vortex_pull_back(&element, p);
            let back = vortex_pull_back(&forward, pulled);
            assert!(distance(back, p) < 2e-4, "{p:?} -> {pulled:?} -> {back:?}");
        }
    }

    #[test]
    fn test_pull_back_preserves_distance_from_the_vortex_line() {
        for circulation in [0.5_f32, 2.0, 6.0, 20.0] {
            let mut element = sample_element();
            element.circulation = circulation;
            let dist = |q: [f32; 3]| {
                let (u, _, v) =
                    vortex_frame_coordinates(&element, vortex_isotropic_offset(&element, q));
                (u * u + v * v).sqrt()
            };
            for p in sample_points() {
                let pulled = vortex_pull_back(&element, p);
                assert!(
                    (dist(pulled) - dist(p)).abs() < 1e-4,
                    "{p:?} changed its distance from the line at circulation {circulation}"
                );
            }
        }
    }

    #[test]
    fn test_pull_back_jvp_matches_finite_differences() {
        let element = sample_element();
        let dirs = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.6, 0.48, -0.64],
            [-0.3, -0.2, 0.9],
        ];
        let eps = 2e-3;
        for p in sample_points() {
            for dir in dirs {
                let (_, jvp) = vortex_pull_back_jvp(&element, p, dir);
                let plus = vortex_pull_back(
                    &element,
                    [
                        p[0] + eps * dir[0],
                        p[1] + eps * dir[1],
                        p[2] + eps * dir[2],
                    ],
                );
                let minus = vortex_pull_back(
                    &element,
                    [
                        p[0] - eps * dir[0],
                        p[1] - eps * dir[1],
                        p[2] - eps * dir[2],
                    ],
                );
                for axis in 0..3 {
                    let numeric = (plus[axis] - minus[axis]) / (2.0 * eps);
                    assert!(
                        (numeric - jvp[axis]).abs() < 5e-3 * (1.0 + jvp[axis].abs()),
                        "p {p:?} dir {dir:?} axis {axis}: fd {numeric} jvp {}",
                        jvp[axis]
                    );
                }
            }
        }
    }

    #[test]
    fn test_composite_pull_back_moves_points_only_while_elements_are_live() {
        let effect = effect_with_branches();
        let field = build_branch_field(&effect, &FlameBaked::default());
        assert!(field.count > 0.0);
        let p = [0.3, 0.6, 0.0];
        let (moved, jvp) = branch_pull_back_jvp(&field, p, [0.0, 0.0, 1.0], effect.time);
        assert!(moved.iter().all(|v| v.is_finite()) && jvp.iter().all(|v| v.is_finite()));
        let after_life = effect.time + field.life + 1.0;
        assert_eq!(branch_pull_back(&field, p, after_life), p);
    }

    #[test]
    fn test_proxy_pad_covers_the_element_balls_and_ignores_gain() {
        let mut effect = effect_with_branches();
        let baked = FlameBaked::default();
        let pad = branch_proxy_pad(&effect, &baked);
        effect.branch.gain = 30.0;
        assert_eq!(branch_proxy_pad(&effect, &baked), pad);

        let trunk = branch_trunk_radius_max(&effect, &baked);
        let ball = effect.branch.reach * (1.0 + effect.branch.spread * BRANCH_SIZE_SCATTER) * trunk;
        let lateral = (effect.branch.core_offset + BRANCH_DRIFT_OVER_LIFE) * trunk;
        assert!((pad.radial - lateral - ball).abs() < 1e-5);
        assert!(pad.top >= 0.0);

        effect.branch.reach *= 2.0;
        assert!(branch_proxy_pad(&effect, &baked).radial > pad.radial);
        effect.branch.period = 0.0;
        assert_eq!(branch_proxy_pad(&effect, &baked), FlameProxyPad::default());
    }

    #[test]
    fn test_core_offset_moves_the_core_to_the_shear_layer_and_widens_the_pad() {
        let mut effect = effect_with_branches();
        let baked = FlameBaked::default();
        let centered = branch_proxy_pad(&effect, &baked);
        effect.branch.core_offset = 1.0;
        let offset = branch_proxy_pad(&effect, &baked);
        let trunk = branch_trunk_radius_max(&effect, &baked);
        assert!((offset.radial - centered.radial - trunk).abs() < 1e-5);

        let field = build_branch_field(&effect, &baked);
        let element = field.elements[0];
        let vortex = vortex_element_at(&field, &element, effect.time).unwrap();
        let age = effect.time - element.spawn_time;
        let lateral =
            (vortex.center[0] * vortex.center[0] + vortex.center[2] * vortex.center[2]).sqrt();
        let expected = element.trunk_radius * (1.0 + field.drift_rate * age);
        assert!((lateral - expected).abs() < 1e-5);
    }

    #[test]
    fn test_reach_scales_the_compact_support_and_keeps_the_growth_ratio() {
        let mut effect = effect_with_branches();
        effect.branch.reach = 2.0;
        let field = build_branch_field(&effect, &FlameBaked::default());
        assert!((field.reach_end - 2.0).abs() < 1e-6);
        assert!((field.reach_start - 2.0 * BRANCH_REACH_GROWTH_START).abs() < 1e-6);
        let element = field.elements[0];
        let vortex = vortex_element_at(&field, &element, effect.time).unwrap();
        let progress = (effect.time - element.spawn_time) / field.life;
        let expected = element.trunk_radius
            * element.size
            * 2.0
            * (BRANCH_REACH_GROWTH_START + (1.0 - BRANCH_REACH_GROWTH_START) * progress);
        assert!((vortex.reach - expected).abs() < 1e-5);
    }

    #[test]
    fn test_envelope_is_zero_at_birth_and_death() {
        let life = 2.0;
        let envelope_time = BRANCH_ENVELOPE_FRACTION * life;
        assert_eq!(branch_envelope(0.0, life, envelope_time), 0.0);
        assert_eq!(branch_envelope(life, life, envelope_time), 0.0);
        let winding_time = BRANCH_WIND_FRACTION * life;
        assert!((branch_envelope(winding_time, life, envelope_time) - 1.0).abs() < 1e-6);
        assert!(
            (branch_envelope(0.7 * life, life, envelope_time) - 1.0).abs() < 1e-6,
            "holds after winding"
        );
        assert!(
            (branch_envelope(0.5 * winding_time, life, envelope_time) - 0.75).abs() < 1e-6,
            "ease-out: three quarters of the angle in the first half of the winding"
        );
        let early = branch_envelope(0.1 * winding_time, life, envelope_time);
        let late = branch_envelope(0.9 * winding_time, life, envelope_time)
            - branch_envelope(0.8 * winding_time, life, envelope_time);
        assert!(early > late, "the medium moves fastest early");
    }

    #[test]
    fn test_burnout_rises_before_the_unwind_and_releases_at_death() {
        let life = 2.0;
        let envelope_time = BRANCH_ENVELOPE_FRACTION * life;
        assert_eq!(branch_burnout(0.0, life, envelope_time), 0.0);
        assert_eq!(
            branch_burnout(BRANCH_BURNOUT_START_FRACTION * life, life, envelope_time),
            0.0
        );
        let unwind_start = life - envelope_time;
        assert!((branch_burnout(unwind_start, life, envelope_time) - 1.0).abs() < 1e-6);
        assert!(
            (branch_burnout(unwind_start + 0.5 * envelope_time, life, envelope_time) - 1.0).abs()
                < 1e-6
        );
        assert_eq!(branch_burnout(life, life, envelope_time), 0.0);
    }

    #[test]
    fn test_burnout_mask_bites_only_the_tongue_outside_the_trunk() {
        let mut element = frame_element([0.8, 0.5, 0.0], 0.0, 0.0, 2.0);
        element.reach = 1.0;
        element.core_radius = 0.5;
        let trunk_radius = 0.8;
        let inside_trunk = [0.3, 0.5, 0.0];
        let tongue = [1.4, 0.5, 0.0];
        let far = [4.0, 0.5, 0.0];
        assert!(
            (vortex_burnout_mask(&element, 1.0, trunk_radius, inside_trunk) - 1.0).abs() < 1e-6
        );
        assert!(vortex_burnout_mask(&element, 1.0, trunk_radius, tongue) < 1e-6);
        assert!((vortex_burnout_mask(&element, 1.0, trunk_radius, far) - 1.0).abs() < 1e-6);
        assert!((vortex_burnout_mask(&element, 0.0, trunk_radius, tongue) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_composite_burnout_mask_is_one_without_burning_elements() {
        let effect = effect_with_branches();
        let field = build_branch_field(&effect, &FlameBaked::default());
        let p = [1.2, 0.5, 0.0];
        let mask = branch_burnout_mask(&field, p, effect.time);
        assert!((0.0..=1.0).contains(&mask));
        let after_life = effect.time + field.life + 1.0;
        assert_eq!(branch_burnout_mask(&field, p, after_life), 1.0);
    }
}
