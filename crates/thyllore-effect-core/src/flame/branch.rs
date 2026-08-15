use super::*;
use std::f32::consts::{PI, TAU};

/// Proxy widening (radial and above the top) that keeps transported density inside
/// the shell cone, in flame-local units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlameProxyPad {
    pub radial: f32,
    pub top: f32,
}

/// One vortex element resolved at a time: the ring core in trunk-local units
/// (y already in aspect-corrected radius units) and its circulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VortexElement {
    pub center: [f32; 3],
    pub arc_center: f32,
    pub arc_half_width: f32,
    pub ring_radius: f32,
    pub core_radius: f32,
    pub circulation: f32,
    pub aspect: f32,
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

fn wrap_angle(angle: f32) -> f32 {
    angle - TAU * (angle / TAU + 0.5).floor()
}

/// Life clamped so at most `BRANCH_MAX_ELEMENTS` elements are ever alive; every
/// element then fades out before it would leave the table.
pub fn effective_branch_life(branch: &FlameBranch) -> f32 {
    let life = branch.life.max(0.0);
    if branch.period <= 0.0 {
        return life;
    }
    life.min((BRANCH_MAX_ELEMENTS as f32 - 1.0) * branch.period)
}

fn spawn_branch_element(branch: &FlameBranch, index: i64) -> FlameBranchElement {
    let spread = branch.spread.clamp(0.0, 1.0);
    let seed = branch.seed;
    let jitter = spread * BRANCH_JITTER_RANGE * (hash01(seed, index, 0) - 0.5);
    let alternating = if index.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
    let side = if hash01(seed, index, 1) < 0.5 * spread {
        -alternating
    } else {
        alternating
    };
    FlameBranchElement {
        spawn_time: (index as f32 + jitter) * branch.period,
        side,
        azimuth: spread * BRANCH_AZIMUTH_RANGE * (hash01(seed, index, 2) - 0.5),
        spawn_height: branch.spawn_height + branch.spawn_range * (hash01(seed, index, 3) - 0.5),
        kind: 0.0,
        hash01: hash01(seed, index, 4),
        _padding: [0.0; 2],
    }
}

/// Elements alive at `time`, newest first; derived from (parameters, time) only.
pub fn active_branch_elements(branch: &FlameBranch, time: f32) -> Vec<FlameBranchElement> {
    if branch.period <= 0.0 || branch.gain == 0.0 {
        return Vec::new();
    }
    let life = effective_branch_life(branch);
    if life <= 0.0 {
        return Vec::new();
    }

    let first = ((time - life) / branch.period).floor() as i64 - 1;
    let last = (time / branch.period).ceil() as i64 + 1;
    let mut elements: Vec<FlameBranchElement> = (first..=last)
        .map(|index| spawn_branch_element(branch, index))
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
    let pattern_scale = effective_noise_aniso_y(effect) * effect.noise_frequency;
    effect.rise_speed / pattern_scale.max(1e-3)
}

pub fn branch_envelope(age: f32, life: f32, envelope_time: f32) -> f32 {
    smoothstep(0.0, envelope_time, age) * (1.0 - smoothstep(life - envelope_time, life, age))
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
    let arc_center = element.azimuth + 0.5 * (1.0 - element.side) * PI;
    let lateral = field.drift_rate * age;
    let center = [
        lateral * arc_center.cos(),
        element.spawn_height + field.rise_rate * age,
        lateral * arc_center.sin(),
    ];
    let progress = age / field.life;
    Some(VortexElement {
        center,
        arc_center,
        arc_half_width: field.arc_half_width,
        ring_radius: field.ring_radius_start
            + (field.ring_radius_end - field.ring_radius_start) * progress,
        core_radius: field.core_radius,
        circulation: field.gain * branch_envelope(age, field.life, field.envelope_time),
        aspect: field.aspect,
    })
}

/// Pull-back through one element with its Jacobian-vector product along `dir`:
/// a rotation of the meridional (radial, axial) plane about the ring core by the
/// windowed Lamb-Oseen angle, compactly supported inside rho < ring_radius.
pub fn vortex_pull_back_jvp(
    element: &VortexElement,
    p: [f32; 3],
    dir: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let center = element.center;
    let aspect = element.aspect;
    let qx = p[0] - center[0];
    let qz = p[2] - center[2];
    let axial = (p[1] - center[1]) * aspect;
    let dx = dir[0];
    let dz = dir[2];
    let d_axial = dir[1] * aspect;

    let dist_sq = qx * qx + qz * qz;
    if dist_sq < 1e-12 {
        return (p, dir);
    }
    let dist = dist_sq.sqrt();
    let inv_dist = 1.0 / dist;
    let ex = qx * inv_dist;
    let ez = qz * inv_dist;
    let ring_radius = element.ring_radius;
    let u = dist - ring_radius;
    let v = axial;
    let rho_sq = u * u + v * v;
    let ring_sq = ring_radius * ring_radius;
    if rho_sq >= ring_sq {
        return (p, dir);
    }
    let x = wrap_angle(qz.atan2(qx) - element.arc_center) / element.arc_half_width;
    if x.abs() >= 1.0 {
        return (p, dir);
    }

    let window = (1.0 - x * x) * (1.0 - x * x);
    let s = rho_sq / ring_sq;
    let gate = (1.0 - s) * (1.0 - s);
    let (profile, d_profile) = lamb_oseen(rho_sq, element.core_radius);
    let circulation = element.circulation;
    let psi = circulation * window * gate * profile;

    let d_dist = ex * dx + ez * dz;
    let dex = (dx - d_dist * ex) * inv_dist;
    let dez = (dz - d_dist * ez) * inv_dist;
    let du = d_dist;
    let dv = d_axial;
    let d_rho_sq = 2.0 * (u * du + v * dv);
    let d_theta = (qx * dz - qz * dx) / dist_sq;
    let d_window = -4.0 * x * (1.0 - x * x) * d_theta / element.arc_half_width;
    let d_gate = -2.0 * (1.0 - s) * d_rho_sq / ring_sq;
    let d_psi = circulation
        * (d_window * gate * profile
            + window * d_gate * profile
            + window * gate * d_profile * d_rho_sq);

    let (sn, cs) = psi.sin_cos();
    let u1 = u * cs - v * sn;
    let v1 = u * sn + v * cs;
    let du1 = du * cs - dv * sn - d_psi * v1;
    let dv1 = du * sn + dv * cs + d_psi * u1;
    let dist1 = ring_radius + u1;
    (
        [
            center[0] + dist1 * ex,
            center[1] + v1 / aspect,
            center[2] + dist1 * ez,
        ],
        [dist1 * dex + du1 * ex, dv1 / aspect, dist1 * dez + du1 * ez],
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

/// Largest chord any point can move under one element: max over rho of
/// 2 rho sin(min(pi, |Gamma| L(rho) g(rho)) / 2), sampled on the disc.
pub fn vortex_reach(circulation: f32, core_radius: f32, ring_radius: f32) -> f32 {
    const SAMPLES: usize = 64;
    (1..SAMPLES)
        .map(|i| {
            let rho = ring_radius * i as f32 / SAMPLES as f32;
            let s = rho * rho / (ring_radius * ring_radius);
            let gate = (1.0 - s) * (1.0 - s);
            let angle = (circulation.abs() * gate * lamb_oseen(rho * rho, core_radius).0).min(PI);
            2.0 * rho * (0.5 * angle).sin()
        })
        .fold(0.0, f32::max)
}

pub fn branch_proxy_pad(effect: &FlameEffect) -> FlameProxyPad {
    let branch = &effect.branch;
    if branch.period <= 0.0 || branch.gain == 0.0 {
        return FlameProxyPad::default();
    }
    let reach = vortex_reach(branch.gain, BRANCH_CORE_RADIUS, BRANCH_RING_RADIUS_END)
        .min(2.0 * BRANCH_RING_RADIUS_END);
    FlameProxyPad {
        radial: BRANCH_DRIFT_OVER_LIFE + reach,
        top: reach / branch_aspect(effect),
    }
}

/// Height over bounding radius: the transport is isotropic in world units, so
/// local y is scaled by this before the meridional rotation.
pub fn branch_aspect(effect: &FlameEffect) -> f32 {
    effect.height.max(MIN_FLAME_EXTENT) / flame_bounding_radius(effect).max(MIN_FLAME_EXTENT)
}

pub fn build_branch_field(effect: &FlameEffect) -> FlameBranchField {
    let branch = &effect.branch;
    let life = effective_branch_life(branch);
    let mut elements = [FlameBranchElement::default(); BRANCH_MAX_ELEMENTS];
    let active = active_branch_elements(branch, effect.time);
    elements[..active.len()].copy_from_slice(&active);
    let pad = branch_proxy_pad(effect);
    FlameBranchField {
        count: active.len() as f32,
        period: branch.period,
        life,
        gain: branch.gain,
        rise_rate: branch_rise_rate(effect),
        drift_rate: BRANCH_DRIFT_OVER_LIFE / life.max(1e-3),
        aspect: branch_aspect(effect),
        core_radius: BRANCH_CORE_RADIUS,
        ring_radius_start: BRANCH_RING_RADIUS_START,
        ring_radius_end: BRANCH_RING_RADIUS_END,
        envelope_time: BRANCH_ENVELOPE_FRACTION * life,
        arc_half_width: BRANCH_ARC_HALF_WIDTH,
        bounding_pad: pad.radial,
        bounding_pad_y: pad.top,
        _padding: [0.0; 2],
        elements,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn branch_on() -> FlameBranch {
        FlameBranch {
            period: 0.4,
            life: 2.5,
            gain: 1.5,
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

    fn sample_element() -> VortexElement {
        VortexElement {
            center: [0.1, 0.5, -0.05],
            arc_center: 0.3,
            arc_half_width: BRANCH_ARC_HALF_WIDTH,
            ring_radius: 0.9,
            core_radius: BRANCH_CORE_RADIUS,
            circulation: 2.0,
            aspect: 2.5,
        }
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
        assert!(active_branch_elements(&branch, 5.0).is_empty());
        let mut branch = branch_on();
        branch.gain = 0.0;
        assert!(active_branch_elements(&branch, 5.0).is_empty());
        let mut effect = FlameEffect::default();
        effect.time = 5.0;
        assert_eq!(build_branch_field(&effect).count, 0.0);
        assert_eq!(branch_proxy_pad(&effect), FlameProxyPad::default());
    }

    #[test]
    fn test_element_table_is_deterministic_and_bounded() {
        let branch = branch_on();
        for time in [0.0_f32, 1.3, 7.77, 123.4] {
            let once = active_branch_elements(&branch, time);
            let twice = active_branch_elements(&branch, time);
            assert_eq!(once, twice);
            assert!(!once.is_empty());
            assert!(once.len() <= BRANCH_MAX_ELEMENTS);
            for pair in once.windows(2) {
                assert!(pair[0].spawn_time > pair[1].spawn_time, "newest first");
            }
            for element in &once {
                let age = time - element.spawn_time;
                assert!(age >= 0.0 && age < effective_branch_life(&branch));
                assert!(element.side == 1.0 || element.side == -1.0);
            }
        }
    }

    #[test]
    fn test_life_clamp_keeps_table_within_capacity_without_dropping() {
        let mut branch = branch_on();
        branch.period = 0.05;
        branch.life = 10.0;
        branch.spread = 1.0;
        let life = effective_branch_life(&branch);
        assert!((life - (BRANCH_MAX_ELEMENTS as f32 - 1.0) * 0.05).abs() < 1e-6);
        let mut step = 0;
        while step < 400 {
            let time = step as f32 * 0.013;
            let elements = active_branch_elements(&branch, time);
            assert!(elements.len() <= BRANCH_MAX_ELEMENTS);
            let alive = ((time - life) / branch.period).floor() as i64
                ..=(time / branch.period).ceil() as i64;
            let unclipped = alive
                .map(|index| spawn_branch_element(&branch, index))
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
    fn test_pull_back_keeps_positive_radial_determinant() {
        for circulation in [0.5_f32, 2.0, 6.0, 20.0] {
            let mut element = sample_element();
            element.circulation = circulation;
            for p in sample_points() {
                let pulled = vortex_pull_back(&element, p);
                let dist = |q: [f32; 3]| {
                    ((q[0] - element.center[0]).powi(2) + (q[2] - element.center[2]).powi(2)).sqrt()
                };
                let before = dist(p);
                let after = dist(pulled);
                if before > 1e-6 && after != before {
                    assert!(
                        after > 0.0,
                        "axis crossed at {p:?} with circulation {circulation}"
                    );
                }
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
        let field = build_branch_field(&effect);
        assert!(field.count > 0.0);
        let p = [0.3, 0.6, 0.0];
        let (moved, jvp) = branch_pull_back_jvp(&field, p, [0.0, 0.0, 1.0], effect.time);
        assert!(moved.iter().all(|v| v.is_finite()) && jvp.iter().all(|v| v.is_finite()));
        let after_life = effect.time + field.life + 1.0;
        assert_eq!(branch_pull_back(&field, p, after_life), p);
    }

    #[test]
    fn test_proxy_pad_grows_with_gain_and_stays_bounded() {
        let mut effect = effect_with_branches();
        let small = branch_proxy_pad(&effect);
        effect.branch.gain = 30.0;
        let large = branch_proxy_pad(&effect);
        assert!(large.radial > small.radial);
        assert!(large.radial <= BRANCH_DRIFT_OVER_LIFE + 2.0 * BRANCH_RING_RADIUS_END + 1e-5);
        assert!(large.top > 0.0);
    }

    #[test]
    fn test_envelope_is_zero_at_birth_and_death() {
        let life = 2.0;
        let envelope_time = BRANCH_ENVELOPE_FRACTION * life;
        assert_eq!(branch_envelope(0.0, life, envelope_time), 0.0);
        assert_eq!(branch_envelope(life, life, envelope_time), 0.0);
        assert!((branch_envelope(1.0, life, envelope_time) - 1.0).abs() < 1e-6);
    }
}
