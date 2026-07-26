use serde::{Deserialize, Serialize};

pub const SCENE_FORMAT_VERSION: u32 = 4;

pub use thyllore_anim_core::editable::{AnimationClipFile, ANIMATION_FORMAT_VERSION};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFile {
    pub version: u32,
    #[serde(default)]
    pub metadata: SceneMetadata,
    pub model: ModelReference,
    #[serde(default)]
    pub animation_clips: Vec<AnimationClipRef>,
    #[serde(default)]
    pub current_clip: Option<String>,
    #[serde(default)]
    pub camera: CameraState,
    #[serde(default)]
    pub timeline: TimelineConfig,
    #[serde(default)]
    pub editor: EditorState,
    #[serde(default)]
    pub panel_layout: Option<PanelLayoutState>,
    #[serde(default)]
    pub flame: Option<FlameSceneData>,
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
            flame: None,
        }
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameSceneData {
    pub effect: FlameEffectData,
    #[serde(default)]
    pub channels: Vec<FlameChannelData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameEffectData {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub height: f32,
    pub radius: f32,
    pub sigma_t: f32,
    pub intensity: f32,
    pub color_base: [f32; 3],
    pub color_tip: [f32; 3],
    pub temperature_base_k: f32,
    pub temperature_tip_k: f32,
    pub use_blackbody: bool,
    pub noise_amplitude: f32,
    pub noise_frequency: f32,
    pub noise_scroll_speed: f32,
    pub time_scale: f32,
    pub time_offset: f32,
    pub warp_amp: f32,
    pub warp_freq: f32,
    pub rise_speed: f32,
    pub taper_power: f32,
    pub radius_tip_ratio: f32,
    pub edge_low: f32,
    pub edge_high: f32,
    pub white_boost: f32,
  pub wind_direction: [f32; 2],
    pub bend_amount: f32,
    pub bend_power: f32,
    pub self_shadow_strength: f32,
    #[serde(default = "default_envelope_peak")]
    pub envelope_peak: f32,
    #[serde(default = "default_envelope_base")]
    pub envelope_base: f32,
    #[serde(default = "default_envelope_tail")]
    pub envelope_tail: f32,
    #[serde(default = "default_radial_sharpness")]
    pub radial_sharpness: f32,
    #[serde(default = "default_occlusion_lum_ref")]
    pub occlusion_lum_ref: f32,
}

fn default_envelope_peak() -> f32 { 0.35 }
fn default_envelope_base() -> f32 { 0.45 }
fn default_envelope_tail() -> f32 { 1.6 }
fn default_radial_sharpness() -> f32 { 4.0 }
fn default_occlusion_lum_ref() -> f32 { 1.0 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameChannelData {
    pub param: String,
    pub keys: Vec<FlameKeyData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameKeyData {
    pub time: f32,
    pub value: f32,
    pub interpolation: String,
}

/// Convert a `FlameParam` variant to its string representation.
pub fn flame_param_to_string(param: crate::ecs::component::FlameParam) -> String {
    match param {
        crate::ecs::component::FlameParam::Height => "Height".to_string(),
        crate::ecs::component::FlameParam::Radius => "Radius".to_string(),
        crate::ecs::component::FlameParam::Intensity => "Intensity".to_string(),
        crate::ecs::component::FlameParam::SigmaT => "SigmaT".to_string(),
        crate::ecs::component::FlameParam::TemperatureBaseK => {
            "TemperatureBaseK".to_string()
        }
        crate::ecs::component::FlameParam::TemperatureTipK => {
            "TemperatureTipK".to_string()
        }
        crate::ecs::component::FlameParam::WarpAmp => "WarpAmp".to_string(),
        crate::ecs::component::FlameParam::WarpFreq => "WarpFreq".to_string(),
        crate::ecs::component::FlameParam::RiseSpeed => "RiseSpeed".to_string(),
        crate::ecs::component::FlameParam::NoiseAmplitude => {
            "NoiseAmplitude".to_string()
        }
        crate::ecs::component::FlameParam::WhiteBoost => "WhiteBoost".to_string(),
        crate::ecs::component::FlameParam::BendAmount => "BendAmount".to_string(),
        crate::ecs::component::FlameParam::WindX => "WindX".to_string(),
        crate::ecs::component::FlameParam::WindZ => "WindZ".to_string(),
        crate::ecs::component::FlameParam::EdgeLow => "EdgeLow".to_string(),
        crate::ecs::component::FlameParam::EdgeHigh => "EdgeHigh".to_string(),
    }
}

/// Convert a string back to a `FlameParam` variant. Unknown strings are ignored (None).
pub fn flame_param_from_string(s: &str) -> Option<crate::ecs::component::FlameParam> {
    match s {
        "Height" => Some(crate::ecs::component::FlameParam::Height),
        "Radius" => Some(crate::ecs::component::FlameParam::Radius),
        "Intensity" => Some(crate::ecs::component::FlameParam::Intensity),
        "SigmaT" => Some(crate::ecs::component::FlameParam::SigmaT),
        "TemperatureBaseK" => {
            Some(crate::ecs::component::FlameParam::TemperatureBaseK)
        }
        "TemperatureTipK" => {
            Some(crate::ecs::component::FlameParam::TemperatureTipK)
        }
        "WarpAmp" => Some(crate::ecs::component::FlameParam::WarpAmp),
        "WarpFreq" => Some(crate::ecs::component::FlameParam::WarpFreq),
        "RiseSpeed" => Some(crate::ecs::component::FlameParam::RiseSpeed),
        "NoiseAmplitude" => {
            Some(crate::ecs::component::FlameParam::NoiseAmplitude)
        }
        "WhiteBoost" => Some(crate::ecs::component::FlameParam::WhiteBoost),
        "BendAmount" => Some(crate::ecs::component::FlameParam::BendAmount),
        "WindX" => Some(crate::ecs::component::FlameParam::WindX),
        "WindZ" => Some(crate::ecs::component::FlameParam::WindZ),
        "EdgeLow" => Some(crate::ecs::component::FlameParam::EdgeLow),
        "EdgeHigh" => Some(crate::ecs::component::FlameParam::EdgeHigh),
        _ => None,
    }
}

/// Convert an `Interpolation` variant to its string representation.
pub fn interpolation_to_string(interp: thyllore_anim_core::Interpolation) -> String {
    match interp {
        thyllore_anim_core::Interpolation::Step => "Step".to_string(),
        thyllore_anim_core::Interpolation::Linear => "Linear".to_string(),
        thyllore_anim_core::Interpolation::CubicSpline => "CubicSpline".to_string(),
    }
}

/// Convert a string back to an `Interpolation` variant. Unknown strings default to Linear.
pub fn interpolation_from_string(s: &str) -> thyllore_anim_core::Interpolation {
    match s {
        "Step" => thyllore_anim_core::Interpolation::Step,
        "Linear" => thyllore_anim_core::Interpolation::Linear,
        "CubicSpline" => thyllore_anim_core::Interpolation::CubicSpline,
        _ => thyllore_anim_core::Interpolation::Linear,
    }
}

/// Build FlameSceneData from the first flame entity's FlameEffect (+FlameTrack if present).
pub fn build_flame_scene_data(world: &crate::ecs::world::World) -> Option<FlameSceneData> {
    let entities: Vec<_> = world.query_flames();
    let entity = entities.first()?;

    let effect = world.get_component::<crate::ecs::resource::FlameEffect>(*entity)?;

    let channels: Vec<FlameChannelData> = if let Some(track) =
        world.get_component::<crate::ecs::component::FlameTrack>(*entity)
    {
        track
            .channels
            .iter()
            .map(|ch| FlameChannelData {
                param: flame_param_to_string(ch.param),
                keys: ch
                    .keys
                    .iter()
                    .map(|k| FlameKeyData {
                        time: k.time,
                        value: k.value,
                        interpolation: interpolation_to_string(k.interpolation.clone()),
                    })
                    .collect(),
            })
            .collect()
    } else {
        Vec::new()
    };

    Some(FlameSceneData {
        effect: FlameEffectData {
            position: [effect.position.x, effect.position.y, effect.position.z],
            rotation: [effect.rotation.s, effect.rotation.v.x, effect.rotation.v.y, effect.rotation.v.z],
            height: effect.height,
            radius: effect.radius,
            sigma_t: effect.sigma_t,
            intensity: effect.intensity,
            color_base: effect.color_base,
            color_tip: effect.color_tip,
            temperature_base_k: effect.temperature_base_k,
            temperature_tip_k: effect.temperature_tip_k,
            use_blackbody: effect.use_blackbody,
            noise_amplitude: effect.noise_amplitude,
            noise_frequency: effect.noise_frequency,
            noise_scroll_speed: effect.noise_scroll_speed,
            time_scale: effect.time_scale,
            time_offset: effect.time_offset,
            warp_amp: effect.warp_amp,
            warp_freq: effect.warp_freq,
            rise_speed: effect.rise_speed,
            taper_power: effect.taper_power,
            radius_tip_ratio: effect.radius_tip_ratio,
            edge_low: effect.edge_low,
            edge_high: effect.edge_high,
            white_boost: effect.white_boost,
            wind_direction: [effect.wind_direction.x, effect.wind_direction.y],
            bend_amount: effect.bend_amount,
            bend_power: effect.bend_power,
            self_shadow_strength: effect.self_shadow_strength,
            envelope_peak: effect.envelope_peak,
            envelope_base: effect.envelope_base,
            envelope_tail: effect.envelope_tail,
            radial_sharpness: effect.radial_sharpness,
            occlusion_lum_ref: effect.occlusion_lum_ref,
        },
        channels,
    })
}

/// Apply loaded flame state to the first flame entity in the world.
pub fn apply_flame_state_to_world(
    world: &mut crate::ecs::world::World,
    flame: &FlameSceneData,
) {
    let entities: Vec<_> = world.query_flames();
    let entity = match entities.first() {
        Some(e) => *e,
        None => return, // no flame entity exists yet, skip silently
    };

    // Write effect fields onto the existing FlameEffect component
    if let Some(mut effect) = world.get_component_mut::<crate::ecs::resource::FlameEffect>(entity) {
        effect.position = cgmath::Vector3::new(flame.effect.position[0], flame.effect.position[1], flame.effect.position[2]);
        effect.rotation = cgmath::Quaternion::new(
            flame.effect.rotation[0],
            flame.effect.rotation[1],
            flame.effect.rotation[2],
            flame.effect.rotation[3],
        );
        effect.height = flame.effect.height;
        effect.radius = flame.effect.radius;
        effect.sigma_t = flame.effect.sigma_t;
        effect.intensity = flame.effect.intensity;
        effect.color_base = flame.effect.color_base;
        effect.color_tip = flame.effect.color_tip;
        effect.temperature_base_k = flame.effect.temperature_base_k;
        effect.temperature_tip_k = flame.effect.temperature_tip_k;
        effect.use_blackbody = flame.effect.use_blackbody;
        effect.noise_amplitude = flame.effect.noise_amplitude;
        effect.noise_frequency = flame.effect.noise_frequency;
        effect.noise_scroll_speed = flame.effect.noise_scroll_speed;
        effect.time_scale = flame.effect.time_scale;
        effect.time_offset = flame.effect.time_offset;
        effect.warp_amp = flame.effect.warp_amp;
        effect.warp_freq = flame.effect.warp_freq;
        effect.rise_speed = flame.effect.rise_speed;
        effect.taper_power = flame.effect.taper_power;
        effect.radius_tip_ratio = flame.effect.radius_tip_ratio;
        effect.edge_low = flame.effect.edge_low;
        effect.edge_high = flame.effect.edge_high;
        effect.white_boost = flame.effect.white_boost;
        effect.wind_direction = cgmath::Vector2::new(flame.effect.wind_direction[0], flame.effect.wind_direction[1]);
        effect.bend_amount = flame.effect.bend_amount;
        effect.bend_power = flame.effect.bend_power;
        effect.self_shadow_strength = flame.effect.self_shadow_strength;
        effect.envelope_peak = flame.effect.envelope_peak;
        effect.envelope_base = flame.effect.envelope_base;
        effect.envelope_tail = flame.effect.envelope_tail;
        effect.radial_sharpness = flame.effect.radial_sharpness;
        effect.occlusion_lum_ref = flame.effect.occlusion_lum_ref;
        thyllore_render_core::refresh_flame_coefficients(&mut effect);
    }

    // Insert/update the FlameTrack component
    let mut channels: Vec<crate::ecs::component::FlameChannel> = Vec::new();
    for ch in &flame.channels {
        let Some(param) = flame_param_from_string(&ch.param) else {
            continue;
        };
        channels.push(crate::ecs::component::FlameChannel {
            param,
            keys: ch
                .keys
                .iter()
                .map(|k| thyllore_anim_core::Keyframe {
                    time: k.time,
                    value: k.value,
                    interpolation: interpolation_from_string(&k.interpolation),
                    in_tangent: None,
                    out_tangent: None,
                })
                .collect(),
        });
    }
    let track = crate::ecs::component::FlameTrack { channels };

    // Remove existing track first (if any) to avoid duplicate
    world.remove_component::<crate::ecs::component::FlameTrack>(entity);
    world.insert_component(entity, track);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flame_param_string_roundtrip() {
        let variants: Vec<crate::ecs::component::FlameParam> = vec![
            crate::ecs::component::FlameParam::Height,
            crate::ecs::component::FlameParam::Radius,
            crate::ecs::component::FlameParam::Intensity,
            crate::ecs::component::FlameParam::SigmaT,
            crate::ecs::component::FlameParam::TemperatureBaseK,
            crate::ecs::component::FlameParam::TemperatureTipK,
            crate::ecs::component::FlameParam::WarpAmp,
            crate::ecs::component::FlameParam::WarpFreq,
            crate::ecs::component::FlameParam::RiseSpeed,
            crate::ecs::component::FlameParam::NoiseAmplitude,
            crate::ecs::component::FlameParam::WhiteBoost,
            crate::ecs::component::FlameParam::BendAmount,
            crate::ecs::component::FlameParam::WindX,
            crate::ecs::component::FlameParam::WindZ,
            crate::ecs::component::FlameParam::EdgeLow,
            crate::ecs::component::FlameParam::EdgeHigh,
        ];
        for v in variants {
            let s = flame_param_to_string(v);
            let roundtrip = flame_param_from_string(&s);
            assert_eq!(roundtrip, Some(v), "Roundtrip failed for {}", s);
        }
    }

    #[test]
    fn test_interpolation_string_roundtrip() {
        let variants: Vec<thyllore_anim_core::Interpolation> = vec![
            thyllore_anim_core::Interpolation::Step,
            thyllore_anim_core::Interpolation::Linear,
            thyllore_anim_core::Interpolation::CubicSpline,
        ];
        for v in variants {
            let s = interpolation_to_string(v.clone());
            let roundtrip = interpolation_from_string(&s);
            assert_eq!(roundtrip, v, "Roundtrip failed for {}", s);
        }
        // Unknown string maps to Linear
        assert_eq!(
            interpolation_from_string("unknown"),
            thyllore_anim_core::Interpolation::Linear
        );
    }

    #[test]
    fn test_flame_scene_data_serde_roundtrip() {
        let scene = FlameSceneData {
            effect: FlameEffectData {
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                height: 1.0,
                radius: 0.5,
                sigma_t: 0.3,
                intensity: 1.0,
                color_base: [1.0, 0.5, 0.0],
                color_tip: [1.0, 1.0, 1.0],
                temperature_base_k: 3200.0,
                temperature_tip_k: 1500.0,
                use_blackbody: true,
                noise_amplitude: 0.1,
                noise_frequency: 1.0,
                noise_scroll_speed: 0.0,
                time_scale: 1.0,
                time_offset: 0.0,
                warp_amp: 0.05,
                warp_freq: 2.0,
                rise_speed: 1.0,
                taper_power: 1.0,
                radius_tip_ratio: 0.10,
                edge_low: 0.3,
                edge_high: 0.7,
                white_boost: 0.0,
                wind_direction: [0.0, 0.0],
                bend_amount: 0.0,
                bend_power: 2.0,
                self_shadow_strength: 0.0,
                envelope_peak: 0.35,
                envelope_base: 0.45,
                envelope_tail: 1.6,
                radial_sharpness: 4.0,
                occlusion_lum_ref: 1.0,
            },
            channels: vec![FlameChannelData {
                param: "Height".to_string(),
                keys: vec![
                    FlameKeyData {
                        time: 0.0,
                        value: 1.0,
                        interpolation: "Linear".to_string(),
                    },
                    FlameKeyData {
                        time: 2.0,
                        value: 2.0,
                        interpolation: "Linear".to_string(),
                    },
                ],
            }],
        };

        let json = serde_json::to_string(&scene).expect("Failed to serialize FlameSceneData");
        let restored: FlameSceneData =
            serde_json::from_str(&json).expect("Failed to deserialize FlameSceneData");

        assert_eq!(scene.effect.position, restored.effect.position);
        assert_eq!(scene.effect.rotation, restored.effect.rotation);
        assert_eq!(scene.effect.height, restored.effect.height);
        assert_eq!(scene.effect.radius, restored.effect.radius);
        assert_eq!(scene.effect.sigma_t, restored.effect.sigma_t);
        assert_eq!(scene.effect.intensity, restored.effect.intensity);
        assert_eq!(scene.effect.color_base, restored.effect.color_base);
        assert_eq!(scene.effect.color_tip, restored.effect.color_tip);
        assert_eq!(scene.effect.temperature_base_k, restored.effect.temperature_base_k);
        assert_eq!(scene.effect.temperature_tip_k, restored.effect.temperature_tip_k);
        assert_eq!(scene.effect.use_blackbody, restored.effect.use_blackbody);
        assert_eq!(scene.effect.noise_amplitude, restored.effect.noise_amplitude);
        assert_eq!(scene.effect.noise_frequency, restored.effect.noise_frequency);
        assert_eq!(scene.effect.noise_scroll_speed, restored.effect.noise_scroll_speed);
        assert_eq!(scene.effect.time_scale, restored.effect.time_scale);
        assert_eq!(scene.effect.time_offset, restored.effect.time_offset);
        assert_eq!(scene.effect.warp_amp, restored.effect.warp_amp);
        assert_eq!(scene.effect.warp_freq, restored.effect.warp_freq);
        assert_eq!(scene.effect.rise_speed, restored.effect.rise_speed);
        assert_eq!(scene.effect.taper_power, restored.effect.taper_power);
        assert_eq!(scene.effect.radius_tip_ratio, restored.effect.radius_tip_ratio);
        assert_eq!(scene.effect.edge_low, restored.effect.edge_low);
        assert_eq!(scene.effect.edge_high, restored.effect.edge_high);
        assert_eq!(scene.effect.white_boost, restored.effect.white_boost);
        assert_eq!(scene.effect.wind_direction, restored.effect.wind_direction);
        assert_eq!(scene.effect.bend_amount, restored.effect.bend_amount);
        assert_eq!(scene.effect.bend_power, restored.effect.bend_power);
        assert_eq!(scene.effect.self_shadow_strength, restored.effect.self_shadow_strength);
        assert_eq!(scene.channels.len(), restored.channels.len());
        assert_eq!(scene.channels[0].param, restored.channels[0].param);
        assert_eq!(scene.channels[0].keys.len(), restored.channels[0].keys.len());
        for (k1, k2) in scene.channels[0].keys.iter().zip(restored.channels[0].keys.iter()) {
            assert_eq!(k1.time, k2.time);
            assert_eq!(k1.value, k2.value);
            assert_eq!(k1.interpolation, k2.interpolation);
        }
    }
}
