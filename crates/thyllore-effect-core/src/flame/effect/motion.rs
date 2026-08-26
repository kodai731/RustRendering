use crate::flame::*;

/// Azimuthal swirl-shear of the RTE medium (differential rotation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameSwirl {
    /// Share of the fixed strain budget spent on the swirl modes; 0 = off.
    pub gain: f32,
    /// Phase-drift rate multiplier of the counter-rotating shear layers.
    pub speed: f32,
}

impl Default for FlameSwirl {
    fn default() -> Self {
        Self {
            gain: 0.0,
            speed: 1.0,
        }
    }
}

/// Node-frozen azimuthal rotation of the noise coordinate (V design).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlameTwist {
    /// Rotation amplitude in radians at the axis tip; rotation never folds, so no strain cap.
    pub gain: f32,
    /// Own phase rate scale; 0 delegates the rate to swirl speed.
    pub speed: f32,
}

/// Animated horizontal displacement of the centerline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameMeander {
    /// Displacement amplitude; 0 = off.
    pub amp: f32,
    /// Multiplier on the meander mode wavenumbers: 1 keeps the two long modes
    /// (kappa 1.2 / 2.1 per height), larger values fold the centerline into a
    /// shorter snake (the pillar reference sits near 12: ~4 bends over the height).
    pub frequency: f32,
}

impl Default for FlameMeander {
    fn default() -> Self {
        Self {
            amp: 0.0,
            frequency: 1.0,
        }
    }
}

/// Sinusoidal displacement of the density boundary; amp 0 = off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameBoundary {
    pub amp: f32,
    pub freq: f32,
    pub speed: f32,
    pub radius_ratio: f32,
}

impl Default for FlameBoundary {
    fn default() -> Self {
        Self {
            amp: 0.2,
            freq: 1.6,
            speed: 0.8,
            radius_ratio: 0.2,
        }
    }
}

/// Discrete branch element layer: a deterministic spawner of vortex transport
/// elements that wrap the RTE medium around a rising core; period 0 or gain 0 = off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameBranch {
    /// Spawn period in seconds; 0 spawns nothing.
    pub period: f32,
    /// Element lifetime in seconds.
    pub life: f32,
    /// Peak rotation angle of the vortex core in radians at full envelope; 0 = off.
    pub gain: f32,
    /// Lamb-Oseen core radius as a ratio of the local trunk radius; near 1 the
    /// whole disc turns coherently (fat tongues), small values shear thin spirals.
    pub core_radius: f32,
    /// Lateral position of the vortex core at spawn as a ratio of the local trunk
    /// radius: 0 sits on the axis (the whole slab tilts), 1 sits on the shear
    /// layer so trunk material rolls outward as a billow.
    pub core_offset: f32,
    /// Compact reach of one element at the end of its life as a ratio of the local
    /// trunk radius; nothing beyond it moves, so it bounds the lateral extent.
    pub reach: f32,
    /// Scatter of azimuth, timing jitter and side alternation in [0, 1].
    pub spread: f32,
    /// Center of the spawn height band in local height units.
    pub spawn_height: f32,
    /// Full width of the spawn height band; 1 with center 0.5 covers the trunk.
    pub spawn_range: f32,
    pub seed: u32,
}

impl Default for FlameBranch {
    fn default() -> Self {
        Self {
            period: 0.0,
            life: 2.5,
            gain: 0.0,
            core_radius: 0.35,
            core_offset: 0.0,
            reach: 1.5,
            spread: 0.3,
            spawn_height: 0.35,
            spawn_range: 0.4,
            seed: 0,
        }
    }
}

/// Puff train: the characteristic solution of the density advection equation
/// along the axis. Parcels of unburnt density leave the base every `period`,
/// rise at `rise`, widen by entrainment and burn out; gain 0 = off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlamePuff {
    /// How far the medium between puffs thins, in [0, 1]: the puff cores keep
    /// the full density, the gaps drop to 1 - gain; 0 = off.
    pub gain: f32,
    /// Spawn period in seconds (puffing frequency 1 / period).
    pub period: f32,
    /// Rise velocity in local height units per second.
    pub rise: f32,
    /// Puff radius at spawn as a ratio of the base trunk radius.
    pub radius: f32,
    /// Radius growth per unit height (entrainment), in spawn radii.
    pub spread: f32,
    /// Height over which the puff density e-folds; 0 = no burnout.
    pub decay: f32,
    /// Vertical over lateral radius of a puff (isotropic units); below 1 = flat lumps.
    pub aspect: f32,
    /// Height at which puffs are spawned, in local height units [0, 1].
    pub spawn_height: f32,
    /// Density of the static root puff; 0 = off.
    pub root_gain: f32,
    /// Center height of the static root puff, in local height units [0, 1].
    pub root_height: f32,
}

impl Default for FlamePuff {
    fn default() -> Self {
        Self {
            gain: 0.0,
            period: 0.5,
            rise: 0.3,
            radius: 0.6,
            spread: 0.5,
            decay: 0.8,
            aspect: 1.0,
            spawn_height: 0.0,
            root_gain: 0.0,
            root_height: 0.0,
        }
    }
}

/// Fluid motion of the column: a Lagrangian marker column (centre and width
/// per height) carried by a 2D vortex-pair flow with a gust, so the silhouette
/// lobes form, deform and sway instead of being advected rigidly; gain 0 = off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameFlow {
    /// Scale of the flow's effect on the column (centre offset and width), 0 = off.
    pub gain: f32,
    /// Vortex pair spawn period in seconds.
    pub period: f32,
    /// Vortex pair rise speed in height units per second.
    pub rise: f32,
    /// Circulation of each vortex in base radii squared per second.
    pub strength: f32,
    /// Gaussian core radius of a vortex in base radii.
    pub core: f32,
    /// Gust velocity amplitude at the tip in base radii per second.
    pub gust: f32,
    /// Base gust frequency in Hz (three incommensurate components around it).
    pub gust_frequency: f32,
    /// Burst (whip) velocity amplitude in base radii per second; 0 = no bursts.
    pub burst: f32,
    /// Restoring rate of the markers toward the rest column, per second.
    pub damping: f32,
    /// Linear reduction of the damping with height: damping at the tip = damping * (1 - damping_slope); 0 = uniform (legacy).
    pub damping_slope: f32,
    /// Upstream transport speed of the marker column in height units per second; 0 = off (bit-match).
    pub transport_speed: f32,
    /// Transport speed increase with height (multiplied by y/aspect); 0 = uniform transport.
    pub transport_accel: f32,
    /// Height01 up to which the gust injects lateral displacement at the root (1 at the foot, 0 at this height); 0 = tip-weighted y/aspect (legacy).
    pub inject_height: f32,
}

impl Default for FlameFlow {
    fn default() -> Self {
        Self {
            gain: 0.0,
            period: 1.0,
            rise: 0.3,
            strength: 1.0,
            core: 0.6,
            gust: 0.3,
            gust_frequency: 0.4,
            burst: 0.0,
            damping: 0.5,
            damping_slope: 0.0,
            transport_speed: 0.0,
            transport_accel: 0.0,
            inject_height: 0.0,
        }
    }
}

/// Lobe train on the silhouette: one-sided bulges that spawn near the foot,
/// rise, swell and fade, riding the flow marker table (needs `flow.gain` > 0);
/// gain 0 = off. Mirrors the round puffs stacked along the reference column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameLobe {
    /// Peak lateral bulge of one lobe in base radii; 0 = off.
    pub gain: f32,
    /// Spawn period in seconds.
    pub period: f32,
    /// Lifetime of one lobe in seconds (swells over the first half, fades over the second).
    pub life: f32,
    /// Rise speed in height units per second.
    pub rise: f32,
    /// Vertical half-extent of one lobe in height units.
    pub size: f32,
    /// Centre of the spawn height band in height units.
    pub spawn_height: f32,
    /// Width of the uniform spawn height band above `spawn_height`; 0 keeps the single band.
    pub spawn_range: f32,
    /// Exponential rise rate in 1/s: the spawn height grows by exp(accel * age), so
    /// higher lobes rise faster (the reference column accelerates with height); 0 = off.
    pub accel: f32,
    /// Scatter of spawn time, height and size in [0, 1].
    pub spread: f32,
    /// Centre shift per unit bulge in [0, 1]: 1 keeps the far side still (a
    /// one-sided tongue), 0 swells both sides (a symmetric puff).
    pub shift: f32,
    /// 1 = inject each lobe once at spawn into the simulated marker column so the flow transport carries it and damping fades it (rise/accel/life unused); 0 = legacy overlay added after the simulation
    pub transport: f32,
}

impl Default for FlameLobe {
    fn default() -> Self {
        Self {
            gain: 0.0,
            period: 0.5,
            life: 2.0,
            rise: 0.1,
            size: 0.08,
            spawn_height: 0.2,
            spawn_range: 0.0,
            accel: 0.0,
            spread: 0.5,
            shift: 1.0,
            transport: 0.0,
        }
    }
}

/// Stateless write-through of the Vortex macro knob onto (twist gain, twist
/// speed); the two parameters stay the single source of truth.
pub fn vortex_macro_parameters(v: f32) -> (f32, f32) {
    let v = v.clamp(0.0, 1.0);
    (VORTEX_MACRO_MAX_GAIN * v, VORTEX_MACRO_MAX_SPEED * v)
}

/// The twist rate scale: twist speed owns the rate when positive, otherwise
/// the rate delegates to the swirl speed.
pub fn twist_rate_scale(twist: &FlameTwist, swirl: &FlameSwirl) -> f32 {
    if twist.speed > 0.0 {
        twist.speed
    } else {
        swirl.speed
    }
}

pub fn build_twist_field(twist: &FlameTwist, swirl: &FlameSwirl) -> FlameTwistField {
    let rate_scale = twist_rate_scale(twist, swirl);
    FlameTwistField {
        modes: std::array::from_fn(|j| FlameTwistMode {
            kappa: TWIST_MODE_KAPPA[j],
            omega: TWIST_MODE_SPIN[j] * rate_scale * twist_mode_phase_rate(TWIST_MODE_KAPPA[j]),
            phase: TWIST_MODE_PHASE[j],
            amp: TWIST_MODE_AMP[j],
        }),
        core_radius_sq: TWIST_CORE_RADIUS_SQ,
        _padding: [0.0; 3],
    }
}

pub fn build_meander_modes(meander: &FlameMeander, swirl: &FlameSwirl) -> [FlameMeanderMode; 2] {
    std::array::from_fn(|j| FlameMeanderMode {
        direction: MEANDER_MODE_DIRECTION[j],
        kappa: MEANDER_MODE_KAPPA[j] * meander.frequency.max(0.0),
        omega: swirl.speed * MEANDER_MODE_RATE_SCALE[j],
        phase: MEANDER_MODE_PHASE[j],
        _padding: [0.0; 3],
    })
}

pub fn build_boundary_params(boundary: &FlameBoundary) -> FlameBoundaryParams {
    FlameBoundaryParams {
        amp: boundary.amp,
        freq: boundary.freq,
        speed: boundary.speed,
        radius_ratio: boundary.radius_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vortex_macro_is_monotone_and_off_at_zero() {
        assert_eq!(vortex_macro_parameters(0.0), (0.0, 0.0));
        assert_eq!(
            vortex_macro_parameters(1.0),
            (VORTEX_MACRO_MAX_GAIN, VORTEX_MACRO_MAX_SPEED)
        );
        assert_eq!(vortex_macro_parameters(2.0), vortex_macro_parameters(1.0));
        assert_eq!(vortex_macro_parameters(-1.0), vortex_macro_parameters(0.0));
        let mut previous = vortex_macro_parameters(0.0);
        for step in 1..=10 {
            let current = vortex_macro_parameters(step as f32 / 10.0);
            assert!(current.0 > previous.0 && current.1 > previous.1);
            previous = current;
        }
    }
}
