use std::sync::OnceLock;

static WAVE_K_RATIO_ENV: OnceLock<f32> = OnceLock::new();
static WAVE_TANH_ENV: OnceLock<f32> = OnceLock::new();

pub(crate) fn read_env_wave_k_ratio() -> f32 {
    *WAVE_K_RATIO_ENV.get_or_init(|| {
        std::env::var("THYLLORE_FLAME_WAVE_K_RATIO")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(crate::flame_wave::WAVE_K_RATIO)
    })
}

pub(crate) fn read_env_wave_tanh() -> f32 {
    *WAVE_TANH_ENV.get_or_init(|| {
        std::env::var("THYLLORE_FLAME_WAVE_TANH")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(crate::flame_wave::WAVE_TANH_SCALE)
    })
}

static WAVE_ENV_MU_ENV: OnceLock<f32> = OnceLock::new();
static WAVE_TRACK_ENV: OnceLock<usize> = OnceLock::new();

/// Tracked erosion mode count (5.1 probabilistic reduction): modes are sorted
/// by |k| ascending and only the first `track` are evaluated per node; the
/// skipped modes' variance enters the erosion response as blur instead.
pub(crate) fn read_env_wave_track() -> usize {
    *WAVE_TRACK_ENV.get_or_init(|| {
        std::env::var("THYLLORE_FLAME_WAVE_TRACK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::flame_wave::WAVE_MODE_COUNT)
            .clamp(1, crate::flame_wave::WAVE_MODE_COUNT)
    })
}

pub(crate) fn read_env_wave_env_mu() -> f32 {
    *WAVE_ENV_MU_ENV.get_or_init(|| {
        std::env::var("THYLLORE_FLAME_WAVE_ENV_MU")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(crate::flame_wave::WAVE_ENV_MU)
    })
}

static WAVE_CF_ENV: OnceLock<bool> = OnceLock::new();
static WAVE_CF_SHEAR_ENV: OnceLock<f32> = OnceLock::new();
static WAVE_CF_LAYERS_ENV: OnceLock<usize> = OnceLock::new();

/// Opt-in switch for the closed-form wave variant (pseudo-FM carriers +
/// single-frequency shear transport), for A/B against the 16-shear baseline.
pub(crate) fn read_env_wave_cf() -> bool {
    *WAVE_CF_ENV.get_or_init(|| {
        std::env::var("THYLLORE_FLAME_WAVE_CF")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

pub(crate) fn read_env_wave_cf_shear() -> f32 {
    *WAVE_CF_SHEAR_ENV.get_or_init(|| {
        std::env::var("THYLLORE_FLAME_WAVE_CF_SHEAR")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(crate::flame_wave::WAVE_CF_SHEAR_GAIN)
    })
}

pub(crate) fn read_env_wave_cf_layers() -> usize {
    *WAVE_CF_LAYERS_ENV.get_or_init(|| {
        std::env::var("THYLLORE_FLAME_WAVE_CF_LAYERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::flame_wave::WAVE_CF_SHEAR_LAYERS)
            .min(crate::flame_wave::WAVE_MODE_SLOTS - crate::flame_wave::WAVE_CF_SHEAR_SLOT)
    })
}

static WARP_FORM_ENV: OnceLock<bool> = OnceLock::new();

/// Warp evaluation form (20260809_warp_asymptotic_strain_redesign.md 追補2):
/// true (default) = one-shot displacement sum whose Jacobian is a bounded SUM,
/// false = the legacy 16-shear sequential composition (multiplicative stretch)
/// for A/B. THYLLORE_FLAME_WARP_FORM = disp | seq.
pub fn read_env_warp_form_displacement() -> bool {
    *WARP_FORM_ENV.get_or_init(|| {
        std::env::var("THYLLORE_FLAME_WARP_FORM")
            .map(|v| !v.eq_ignore_ascii_case("seq"))
            .unwrap_or(true)
    })
}

static SWIRL_GAIN_ENV: OnceLock<Option<f32>> = OnceLock::new();

/// Calibration-only override of FlameEffect::swirl_gain (motion_design L2);
/// removed once the adopted value is baked into the presets.
pub fn read_env_swirl_gain(effect_value: f32) -> f32 {
    SWIRL_GAIN_ENV
        .get_or_init(|| {
            std::env::var("THYLLORE_FLAME_SWIRL_GAIN")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(effect_value)
}

static WAVE_UNIFIED_ENV: OnceLock<bool> = OnceLock::new();

/// Unified broadband field (20260809_unified_field_redesign.md): one 128-mode
/// gap-free table replaces boundary fbm / contour wiggle / phase jitter, and
/// the response window gains a modulation-proportional sigma floor.
/// THYLLORE_FLAME_UNIFIED = 1 (default) | 0 = legacy path for A/B.
pub fn read_env_wave_unified() -> bool {
    *WAVE_UNIFIED_ENV.get_or_init(|| {
        std::env::var("THYLLORE_FLAME_UNIFIED")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true)
    })
}

/// Parameter mapping of the unified spectral tilt: old boundary_amp / wiggle_amp
/// become smooth low- / mid-octave gains of the one table.
pub const UNIFIED_BOUNDARY_TILT_GAIN: f32 = 10.0;
pub const UNIFIED_WIGGLE_TILT_GAIN: f32 = 2.0;
/// Relative response window: sigma floor = beta * local modulation std.
pub const UNIFIED_WINDOW_BETA: f32 = 0.75;

static UNIFIED_BETA_ENV: OnceLock<f32> = OnceLock::new();
static UNIFIED_TILT_B_ENV: OnceLock<f32> = OnceLock::new();
static UNIFIED_TILT_W_ENV: OnceLock<f32> = OnceLock::new();

pub fn read_env_unified_beta() -> f32 {
    *UNIFIED_BETA_ENV.get_or_init(|| {
        std::env::var("THYLLORE_FLAME_UNIFIED_BETA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(UNIFIED_WINDOW_BETA)
    })
}

pub fn read_env_unified_tilt_gain_b() -> f32 {
    *UNIFIED_TILT_B_ENV.get_or_init(|| {
        std::env::var("THYLLORE_FLAME_UNIFIED_TILT_B")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(UNIFIED_BOUNDARY_TILT_GAIN)
    })
}

pub fn read_env_unified_tilt_gain_w() -> f32 {
    *UNIFIED_TILT_W_ENV.get_or_init(|| {
        std::env::var("THYLLORE_FLAME_UNIFIED_TILT_W")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(UNIFIED_WIGGLE_TILT_GAIN)
    })
}

static WAVE_MODE_MASK_ENV: OnceLock<Option<String>> = OnceLock::new();

pub(crate) fn read_env_wave_mode_mask() -> Option<String> {
    WAVE_MODE_MASK_ENV
        .get_or_init(|| std::env::var("THYLLORE_FLAME_WAVE_MODE_MASK").ok())
        .clone()
}

pub const NOISE_AMPLITUDE_REF: f32 = 1.5;
pub const EDGE_WIDTH_GAMMA: f32 = 1.0;
pub const SHAPING_GAMMA: f32 = 0.5;
