pub(crate) const MIN_FLAME_EXTENT: f32 = 1e-3;

/// Vortex macro (plan D): one UI knob in [0, 1] mapped onto both twist
/// parameters along a monotone "faster and deeper" curve.
pub const VORTEX_MACRO_MAX_GAIN: f32 = 6.0;
pub const VORTEX_MACRO_MAX_SPEED: f32 = 2.0;

/// Noise Sharpness macro endpoints: the tanh shaping scale at knob 0 (soft)
/// and knob 1 (sharp), remapped along a log curve.
pub const NOISE_SHARPNESS_SCALE_SOFT: f32 = 6.0;
pub const NOISE_SHARPNESS_SCALE_SHARP: f32 = 0.1;

/// Medium twist field (V design): Lamb-Oseen core radius^2 and the two
/// counter-rotating axial modes, sent to the shader through the UBO.
pub const TWIST_CORE_RADIUS_SQ: f32 = 0.49;
pub const TWIST_MODE_KAPPA: [f32; 2] = [2.2, 3.6];
pub const TWIST_MODE_PHASE: [f32; 2] = [0.7, 2.9];
pub const TWIST_MODE_AMP: [f32; 2] = [0.65, 0.35];
/// Rotation direction per mode: the two depth layers swirl against each other.
pub const TWIST_MODE_SPIN: [f32; 2] = [-1.0, 1.0];

/// Eddy-turnover phase rate: omega_j = rate_scale * (kappa_j / 2pi)^(2/3).
pub fn twist_mode_phase_rate(kappa: f32) -> f32 {
    (kappa / std::f32::consts::TAU).powf(2.0 / 3.0)
}

/// Animated meander modes: two horizontal sinusoids with frequencies derived
/// from swirl speed, sent to the shader through the UBO.
pub const MEANDER_MODE_DIRECTION: [[f32; 2]; 2] = [[1.0, 0.0], [0.6, 0.8]];
pub const MEANDER_MODE_KAPPA: [f32; 2] = [1.2, 2.1];
pub const MEANDER_MODE_PHASE: [f32; 2] = [0.0, 2.4];
pub const MEANDER_MODE_RATE_SCALE: [f32; 2] = [0.75, 1.15];

/// Branch element layer (flame_branch_elements design): element table size and
/// the age-profile constants of the vortex transport, in trunk-local radius units.
pub const BRANCH_MAX_ELEMENTS: usize = 32;
/// Reach at spawn as a ratio of the element's final reach (entrainment growth).
pub const BRANCH_REACH_GROWTH_START: f32 = 0.6;
pub const BRANCH_DRIFT_OVER_LIFE: f32 = 0.5;
pub const BRANCH_ENVELOPE_FRACTION: f32 = 0.15;
/// Age fraction where the tongue starts burning out (density fade outside the trunk).
pub const BRANCH_BURNOUT_START_FRACTION: f32 = 0.5;
/// Fraction of the unwind window over which the burnout mask releases, once the
/// remaining rotation is negligible.
pub const BRANCH_BURNOUT_RELEASE_FRACTION: f32 = 0.1;
/// Burnout plateau extends this ratio beyond the element reach before fading out.
pub const BRANCH_BURNOUT_MARGIN: f32 = 0.5;
/// Trunk radius ratio inside which the burnout never touches the medium.
pub const BRANCH_BURNOUT_TRUNK_INNER: f32 = 0.75;
pub const BRANCH_AZIMUTH_RANGE: f32 = std::f32::consts::PI;
/// Spawn-time jitter range as a fraction of the period; below 1 keeps spawn order.
pub const BRANCH_JITTER_RANGE: f32 = 0.5;
