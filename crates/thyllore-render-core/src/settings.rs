#[derive(Clone, Debug)]
pub struct BloomSettings {
    pub enabled: bool,
    pub intensity: f32,
    pub threshold: f32,
    pub knee: f32,
    pub mip_count: u32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.04,
            threshold: 1.0,
            knee: 0.5,
            mip_count: 5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DepthOfField {
    pub enabled: bool,
    pub focus_distance: f32,
    pub max_blur_radius: f32,
}

impl Default for DepthOfField {
    fn default() -> Self {
        Self {
            enabled: false,
            focus_distance: 10.0,
            max_blur_radius: 8.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(i32)]
pub enum ToneMapOperator {
    None = 0,
    AcesFilmic = 1,
    Reinhard = 2,
}

#[derive(Clone, Debug)]
pub struct ToneMapping {
    pub enabled: bool,
    pub operator: ToneMapOperator,
    pub gamma: f32,
}

impl Default for ToneMapping {
    fn default() -> Self {
        Self {
            enabled: true,
            operator: ToneMapOperator::AcesFilmic,
            gamma: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Exposure {
    pub ev100: f32,
    pub exposure_value: f32,
}

impl Default for Exposure {
    fn default() -> Self {
        Self {
            ev100: -0.263,
            exposure_value: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LensEffects {
    pub vignette_enabled: bool,
    pub vignette_intensity: f32,
    pub chromatic_aberration_enabled: bool,
    pub chromatic_aberration_intensity: f32,
}

impl Default for LensEffects {
    fn default() -> Self {
        Self {
            vignette_enabled: false,
            vignette_intensity: 0.3,
            chromatic_aberration_enabled: false,
            chromatic_aberration_intensity: 0.005,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AutoExposure {
    pub enabled: bool,
    pub min_ev: f32,
    pub max_ev: f32,
    pub adaptation_speed_up: f32,
    pub adaptation_speed_down: f32,
    pub low_percent: f32,
    pub high_percent: f32,
    pub min_log_luminance: f32,
    pub log_luminance_range: f32,
    pub saved_manual_exposure: Option<f32>,
}

impl Default for AutoExposure {
    fn default() -> Self {
        Self {
            enabled: true,
            min_ev: 0.0,
            max_ev: 12.0,
            adaptation_speed_up: 3.0,
            adaptation_speed_down: 1.0,
            low_percent: 0.1,
            high_percent: 0.9,
            min_log_luminance: -10.0,
            log_luminance_range: 22.0,
            saved_manual_exposure: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PhysicalCameraParameters {
    pub focal_length_mm: f32,
    pub sensor_height_mm: f32,
    pub aperture_f_stops: f32,
    pub shutter_speed_s: f32,
    pub sensitivity_iso: f32,
}

impl Default for PhysicalCameraParameters {
    fn default() -> Self {
        Self {
            focal_length_mm: 35.0,
            sensor_height_mm: 18.66,
            aperture_f_stops: 16.0,
            shutter_speed_s: 1.0 / 125.0,
            sensitivity_iso: 100.0,
        }
    }
}
