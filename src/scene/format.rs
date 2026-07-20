use serde::{Deserialize, Serialize};

pub const SCENE_FORMAT_VERSION: u32 = 4;

pub use thyllore_anim_core::editable::{AnimationClipFile, ANIMATION_FORMAT_VERSION};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFile {
    pub version: u32,
    pub metadata: SceneMetadata,
    pub model: ModelReference,
    pub animation_clips: Vec<AnimationClipRef>,
    pub current_clip: Option<String>,
    pub camera: CameraState,
    pub timeline: TimelineConfig,
    pub editor: EditorState,
    #[serde(default)]
    pub panel_layout: Option<PanelLayoutState>,
}

impl SceneFile {
    pub fn new(name: &str, model_path: &str) -> Self {
        Self {
            version: SCENE_FORMAT_VERSION,
            metadata: SceneMetadata::new(name),
            model: ModelReference::new(model_path),
            animation_clips: Vec::new(),
            current_clip: None,
            camera: CameraState::default(),
            timeline: TimelineConfig::default(),
            editor: EditorState::default(),
            panel_layout: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneMetadata {
    pub name: String,
    pub created_at: String,
    pub modified_at: String,
}

impl SceneMetadata {
    pub fn new(name: &str) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            name: name.to_string(),
            created_at: now.clone(),
            modified_at: now,
        }
    }

    pub fn update_modified(&mut self) {
        self.modified_at = chrono::Utc::now().to_rfc3339();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelReference {
    pub path: String,
    pub transform: TransformData,
}

impl ModelReference {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            transform: TransformData::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformData {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl Default for TransformData {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationClipRef {
    pub path: String,
}

impl AnimationClipRef {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraState {
    pub pivot: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub fov_y: f32,

    #[serde(default)]
    pub position: Option<[f32; 3]>,
    #[serde(default)]
    pub direction: Option<[f32; 3]>,
    #[serde(default)]
    pub up: Option<[f32; 3]>,

    #[serde(default)]
    pub physical_camera: Option<PhysicalCameraState>,
    #[serde(default)]
    pub exposure: Option<ExposureState>,
    #[serde(default)]
    pub depth_of_field: Option<DepthOfFieldState>,
    #[serde(default)]
    pub tone_mapping: Option<ToneMappingState>,
    #[serde(default)]
    pub lens_effects: Option<LensEffectsState>,
    #[serde(default)]
    pub bloom: Option<BloomState>,
    #[serde(default)]
    pub auto_exposure: Option<AutoExposureState>,
}

impl Default for CameraState {
    fn default() -> Self {
        use std::f32::consts::PI;
        Self {
            pivot: [0.0, 0.0, 0.0],
            yaw: PI / 4.0,
            pitch: (5.0_f32 / 75.0_f32.sqrt()).asin(),
            distance: 75.0_f32.sqrt(),
            fov_y: 45.0,
            position: None,
            direction: None,
            up: None,
            physical_camera: None,
            exposure: None,
            depth_of_field: None,
            tone_mapping: None,
            lens_effects: None,
            bloom: None,
            auto_exposure: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineConfig {
    pub current_time: f32,
    pub playing: bool,
    pub looping: bool,
    pub speed: f32,
}

impl Default for TimelineConfig {
    fn default() -> Self {
        Self {
            current_time: 0.0,
            playing: false,
            looping: true,
            speed: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorState {
    pub selected_bone_id: Option<u32>,
    pub curve_editor_open: bool,
}

impl Default for EditorState {
    fn default() -> Self {
        Self {
            selected_bone_id: None,
            curve_editor_open: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelLayoutState {
    pub hierarchy_width: f32,
    pub inspector_width: f32,
    pub timeline_height: f32,
    pub debug_height: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalCameraState {
    pub focal_length_mm: f32,
    pub sensor_height_mm: f32,
    pub aperture_f_stops: f32,
    pub shutter_speed_s: f32,
    pub sensitivity_iso: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureState {
    pub ev100: f32,
    pub exposure_value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthOfFieldState {
    pub enabled: bool,
    pub focus_distance: f32,
    pub max_blur_radius: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToneMappingState {
    pub enabled: bool,
    pub operator: String,
    pub gamma: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensEffectsState {
    pub vignette_enabled: bool,
    pub vignette_intensity: f32,
    pub chromatic_aberration_enabled: bool,
    pub chromatic_aberration_intensity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomState {
    pub enabled: bool,
    pub intensity: f32,
    pub threshold: f32,
    pub knee: f32,
    pub mip_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoExposureState {
    pub enabled: bool,
    pub min_ev: f32,
    pub max_ev: f32,
    pub adaptation_speed_up: f32,
    pub adaptation_speed_down: f32,
    pub low_percent: f32,
    pub high_percent: f32,
}
