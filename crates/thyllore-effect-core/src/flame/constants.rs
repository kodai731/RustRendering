pub(crate) const MIN_FLAME_EXTENT: f32 = 1e-3;

/// Vortex macro (plan D): one UI knob in [0, 1] mapped onto both twist
/// parameters along a monotone "faster and deeper" curve.
pub const VORTEX_MACRO_MAX_GAIN: f32 = 6.0;
pub const VORTEX_MACRO_MAX_SPEED: f32 = 2.0;

/// Noise Sharpness macro endpoints: the tanh shaping scale at knob 0 (soft)
/// and knob 1 (sharp), remapped along a log curve.
pub const NOISE_SHARPNESS_SCALE_SOFT: f32 = 6.0;
pub const NOISE_SHARPNESS_SCALE_SHARP: f32 = 0.1;

/// Runtime range of the closed-form wave walk segment count per ray.
pub const WAVE_SEGMENTS_MIN: u32 = 16;
pub const WAVE_SEGMENTS_MAX: u32 = 256;

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
pub const MEANDER_MODE_DIRECTION: [[f32; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];
pub const MEANDER_MODE_KAPPA: [f32; 2] = [1.2, 2.1];
pub const MEANDER_MODE_PHASE: [f32; 2] = [0.0, 2.4];
pub const MEANDER_MODE_RATE_SCALE: [f32; 2] = [0.75, 1.15];

/// Branch element layer (flame_branch_elements design): element table size and
/// the age-profile constants of the vortex transport, in trunk-local radius units.
pub const BRANCH_MAX_ELEMENTS: usize = 32;
/// Puff train: table size and the spawn-time jitter as a fraction of the period.
pub const PUFF_MAX_COUNT: usize = 16;
pub const PUFF_SPAWN_JITTER: f32 = 0.3;
/// Fluid motion: marker column size, vortex table size, the coarsest
/// integration step and the longest history window of the stateless re-simulation.
pub const GRID_WIDTH_CELLS: usize = 64;
pub const GRID_HEIGHT_CELLS: usize = 128;
pub const GRID_HEIGHT_EXTENT: f32 = 1.1;
pub const GRID_CFL_LIMIT: f32 = 0.8;
pub const GRID_BORDER_FADE_CELLS: f32 = 3.0;

pub const FLOW_MARKER_COUNT: usize = 32;
pub const FLOW_VORTEX_MAX_PAIRS: usize = 16;
pub const FLOW_SIM_DT: f32 = 1.0 / 60.0;
pub const FLOW_HISTORY_SECONDS: f32 = 12.0;
pub const FLOW_SPAWN_JITTER: f32 = 0.3;
/// Reach at spawn as a ratio of the element's final reach (entrainment growth).
pub const BRANCH_REACH_GROWTH_START: f32 = 0.6;
pub const BRANCH_DRIFT_OVER_LIFE: f32 = 0.5;
pub const BRANCH_ENVELOPE_FRACTION: f32 = 0.15;
/// Age fraction by which the core angle has fully wound (ease-out from birth).
pub const BRANCH_WIND_FRACTION: f32 = 0.5;
/// Age fraction where the tongue starts burning out (density fade outside the
/// trunk); after the winding is complete so the tongue is seen at full extent.
pub const BRANCH_BURNOUT_START_FRACTION: f32 = 0.6;
/// Fraction of the unwind window over which the burnout mask releases, once the
/// remaining rotation is negligible.
pub const BRANCH_BURNOUT_RELEASE_FRACTION: f32 = 0.1;
/// Burnout plateau extends this ratio beyond the element reach before fading out.
pub const BRANCH_BURNOUT_MARGIN: f32 = 0.5;
/// Trunk radius ratio inside which the burnout never touches the medium.
pub const BRANCH_BURNOUT_TRUNK_INNER: f32 = 0.75;
/// Azimuth step between consecutive tongues: the golden angle 2pi(1 - 1/phi) fills
/// the circle uniformly without periodic alignment.
pub const BRANCH_AZIMUTH_GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;
/// Full-spread jitter range around an element's azimuth slot.
pub const BRANCH_AZIMUTH_JITTER: f32 = std::f32::consts::PI;
/// Spawn-time jitter range as a fraction of the period; below 1 keeps spawn order.
pub const BRANCH_JITTER_RANGE: f32 = 0.5;
/// Per-element scatter driven by `spread`: size multiplier range (+-), line tilt
/// out of the horizontal [rad], and window center shift along the line in reach
/// units. Together they keep the tongues from reading as identical rotated slabs.
pub const BRANCH_SIZE_SCATTER: f32 = 0.5;
pub const BRANCH_TILT_RANGE: f32 = 0.5;
pub const BRANCH_ALONG_OFFSET_RANGE: f32 = 0.5;
