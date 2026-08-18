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
