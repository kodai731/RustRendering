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
    /// Written in place of a file path when the mesh was generated in-app.
    pub const GENERATED_MESH: &'static str = "Generated Mesh";

    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            transform: TransformData::default(),
        }
    }

    pub fn is_generated_mesh(&self) -> bool {
        self.path == Self::GENERATED_MESH
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
    #[serde(default)]
    pub motion_path: Option<MotionPathData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionPathData {
    pub center: [f32; 3],
    pub radius: f32,
    pub angular_speed: f32,
    pub phase_offset: f32,
    pub enabled: bool,
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
    #[serde(default = "default_contour_wiggle_amp")]
    pub contour_wiggle_amp: f32,
    #[serde(default)]
    pub aniso_axis_advect: f32,
    #[serde(default = "default_rte_bands")]
    pub rte_bands: f32,
    #[serde(default = "default_sigma_dispersion")]
    pub sigma_dispersion: f32,
    #[serde(default)]
    pub edge_temperature_blend: f32,
    #[serde(default = "default_tip_carve_depth")]
    pub tip_carve_depth: f32,
    #[serde(default = "default_tip_carve_reach")]
    pub tip_carve_reach: f32,
    #[serde(default = "default_warp_reach")]
    pub warp_reach: f32,
}

fn default_tip_carve_depth() -> f32 {
    1.0
}
fn default_tip_carve_reach() -> f32 {
    0.2
}
fn default_warp_reach() -> f32 {
    thyllore_render_core::flame_wave::WARP_REACH_DEFAULT
}

fn default_rte_bands() -> f32 {
    4.0
}
fn default_sigma_dispersion() -> f32 {
    1.0
}

fn default_envelope_peak() -> f32 {
    0.35
}
fn default_envelope_base() -> f32 {
    0.45
}
fn default_envelope_tail() -> f32 {
    1.6
}
fn default_radial_sharpness() -> f32 {
    4.0
}
fn default_occlusion_lum_ref() -> f32 {
    1.0
}
fn default_contour_wiggle_amp() -> f32 {
    0.3
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameChannelData {
    pub param: String,
    pub keys: Vec<FlameKeyData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameKeyData {
    pub time: f32,
    pub value: f32,
    #[serde(default = "default_flame_interpolation")]
    pub interpolation: String,
    pub in_tangent: Option<[f32; 2]>,
    pub out_tangent: Option<[f32; 2]>,
    pub weight_mode: Option<String>,
}

fn default_flame_interpolation() -> String {
    "Linear".to_string()
}

/// Convert an `Interpolation` variant to its string representation.
pub fn interpolation_to_string(interp: thyllore_anim_core::Interpolation) -> String {
    match interp {
        thyllore_anim_core::Interpolation::Step => "Step".to_string(),
        thyllore_anim_core::Interpolation::Linear => "Linear".to_string(),
        thyllore_anim_core::Interpolation::CubicSpline => "CubicSpline".to_string(),
    }
}

/// Convert an editable `InterpolationType` variant to its string representation.
fn editable_interpolation_to_string(
    interp: thyllore_anim_core::editable::InterpolationType,
) -> String {
    match interp {
        thyllore_anim_core::editable::InterpolationType::Stepped => "Step".to_string(),
        thyllore_anim_core::editable::InterpolationType::Linear => "Linear".to_string(),
        thyllore_anim_core::editable::InterpolationType::Bezier => "Bezier".to_string(),
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

/// Build FlameSceneData from the first flame entity's FlameEffect, with keyframe
/// channels read from the flame's scheduled clip (scalar curves). The on-disk
/// FlameChannelData format is unchanged, so pre-clip scenes stay compatible.
pub fn build_flame_scene_data(world: &crate::ecs::world::World) -> Option<FlameSceneData> {
    let entities: Vec<_> = world.query_flames();
    let entity = entities.first()?;

    let effect = world.get_component::<crate::ecs::component::FlameEffect>(*entity)?;

    let channels: Vec<FlameChannelData> = build_flame_channels_from_clip(world, *entity);

    let motion_path = world
        .get_component::<crate::ecs::component::MotionPath>(*entity)
        .map(|mp| MotionPathData {
            center: [mp.center.x, mp.center.y, mp.center.z],
            radius: mp.radius,
            angular_speed: mp.angular_speed,
            phase_offset: mp.phase_offset,
            enabled: mp.enabled,
        });

    Some(FlameSceneData {
        effect: FlameEffectData {
            position: [effect.position.x, effect.position.y, effect.position.z],
            rotation: [
                effect.rotation.s,
                effect.rotation.v.x,
                effect.rotation.v.y,
                effect.rotation.v.z,
            ],
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
            contour_wiggle_amp: effect.contour_wiggle_amp,
            aniso_axis_advect: effect.aniso_axis_advect,
            rte_bands: effect.rte_bands,
            sigma_dispersion: effect.sigma_dispersion,
            tip_carve_depth: effect.tip_carve_depth,
            tip_carve_reach: effect.tip_carve_reach,
            warp_reach: effect.warp_reach,
            edge_temperature_blend: effect.edge_temperature_blend,
        },
        channels,
        motion_path,
    })
}

fn build_flame_channels_from_clip(
    world: &crate::ecs::world::World,
    entity: crate::ecs::world::Entity,
) -> Vec<FlameChannelData> {
    let Some(clip_id) = crate::ecs::systems::find_entity_clip_id(world, entity) else {
        return Vec::new();
    };
    let Some(lib) = world.get_resource::<crate::ecs::resource::ClipLibrary>() else {
        return Vec::new();
    };
    let Some(clip) = lib.get(clip_id) else {
        return Vec::new();
    };

    clip.scalar_curves
        .iter()
        .filter_map(|curve| {
            let (_, channel) =
                crate::ecs::component::scalar_channel_for_property(curve.property_type)?;
            Some(FlameChannelData {
                param: channel.scene_name.to_string(),
                keys: curve
                    .keyframes
                    .iter()
                    .map(|k| FlameKeyData {
                        time: k.time,
                        value: k.value,
                        interpolation: editable_interpolation_to_string(k.interpolation),
                        in_tangent: Some([k.in_tangent.time_offset, k.in_tangent.value_offset]),
                        out_tangent: Some([k.out_tangent.time_offset, k.out_tangent.value_offset]),
                        weight_mode: Some(match k.weight_mode {
                            thyllore_anim_core::editable::TangentWeightMode::NonWeighted => {
                                "NonWeighted".to_string()
                            }
                            thyllore_anim_core::editable::TangentWeightMode::Weighted => {
                                "Weighted".to_string()
                            }
                        }),
                    })
                    .collect(),
            })
        })
        .collect()
}

/// Apply loaded flame state to the first flame entity in the world.
pub fn apply_flame_state_to_world(
    world: &mut crate::ecs::world::World,
    assets: &mut crate::asset::AssetStorage,
    flame: &FlameSceneData,
) {
    let entities: Vec<_> = world.query_flames();
    let entity = match entities.first() {
        Some(e) => *e,
        None => return, // no flame entity exists yet, skip silently
    };

    // Write effect fields onto the existing FlameEffect component
    if let Some(mut effect) = world.get_component_mut::<crate::ecs::component::FlameEffect>(entity)
    {
        effect.position = cgmath::Vector3::new(
            flame.effect.position[0],
            flame.effect.position[1],
            flame.effect.position[2],
        );
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
        effect.wind_direction = cgmath::Vector2::new(
            flame.effect.wind_direction[0],
            flame.effect.wind_direction[1],
        );
        effect.bend_amount = flame.effect.bend_amount;
        effect.bend_power = flame.effect.bend_power;
        effect.self_shadow_strength = flame.effect.self_shadow_strength;
        effect.envelope_peak = flame.effect.envelope_peak;
        effect.envelope_base = flame.effect.envelope_base;
        effect.envelope_tail = flame.effect.envelope_tail;
        effect.occlusion_lum_ref = flame.effect.occlusion_lum_ref;
        effect.contour_wiggle_amp = flame.effect.contour_wiggle_amp;
        effect.aniso_axis_advect = flame.effect.aniso_axis_advect;
        effect.rte_bands = flame.effect.rte_bands;
        effect.sigma_dispersion = flame.effect.sigma_dispersion;
        effect.tip_carve_depth = flame.effect.tip_carve_depth;
        effect.tip_carve_reach = flame.effect.tip_carve_reach;
        effect.warp_reach = flame.effect.warp_reach;
        thyllore_render_core::refresh_flame_coefficients(&mut effect);
    }

    crate::ecs::systems::write_flame_transform(
        world,
        entity,
        cgmath::Vector3::new(
            flame.effect.position[0],
            flame.effect.position[1],
            flame.effect.position[2],
        ),
        cgmath::Quaternion::new(
            flame.effect.rotation[0],
            flame.effect.rotation[1],
            flame.effect.rotation[2],
            flame.effect.rotation[3],
        ),
    );

    // Rebuild the flame clip (scalar curves) from the scene channels. Loading is
    // idempotent: any previously scheduled flame clip instance is replaced.
    let mut editable = thyllore_anim_core::editable::EditableAnimationClip::new(
        0,
        crate::ecs::component::FLAME_DOMAIN.name.to_string(),
    );
    for ch in &flame.channels {
        let Some((_, channel)) = crate::ecs::component::scalar_channel_for_scene_name(&ch.param)
        else {
            continue;
        };
        let curve = editable.get_or_add_scalar_curve(channel.property_type());
        for k in ch.keys.iter() {
            let interp = match k.interpolation.as_str() {
                "Bezier" => thyllore_anim_core::editable::InterpolationType::Bezier,
                "Step" => thyllore_anim_core::editable::InterpolationType::Stepped,
                _ => thyllore_anim_core::editable::InterpolationType::Linear,
            };
            let id = thyllore_anim_core::editable::curve_add_keyframe(curve, k.time, k.value);
            let key = curve.get_keyframe_mut(id).expect("key just inserted");
            key.interpolation = interp;
            if let Some([t, v]) = k.in_tangent {
                key.in_tangent.time_offset = t;
                key.in_tangent.value_offset = v;
            }
            if let Some([t, v]) = k.out_tangent {
                key.out_tangent.time_offset = t;
                key.out_tangent.value_offset = v;
            }
            if let Some(wm) = &k.weight_mode {
                key.weight_mode = match wm.as_str() {
                    "Weighted" => thyllore_anim_core::editable::TangentWeightMode::Weighted,
                    _ => thyllore_anim_core::editable::TangentWeightMode::NonWeighted,
                };
            }
        }
    }
    thyllore_anim_core::editable::clip_recalculate_duration(&mut editable);

    world.remove_component::<crate::ecs::component::ClipSchedule>(entity);
    if editable.has_scalar_keyframes() {
        let clip_id = {
            let mut clip_library = world.resource_mut::<crate::ecs::resource::ClipLibrary>();
            crate::ecs::systems::clip_library_systems::clip_library_register_and_activate(
                &mut clip_library,
                assets,
                editable,
            )
        };
        let mut schedule = crate::ecs::component::ClipSchedule::new();
        let instance_id = schedule.next_instance_id;
        schedule.next_instance_id += 1;
        schedule
            .instances
            .push(thyllore_anim_core::editable::ClipInstance::new(
                instance_id,
                clip_id,
                0.0,
            ));
        world.insert_component(entity, schedule);
    }

    // Insert MotionPath component if present in scene data
    if let Some(mp) = &flame.motion_path {
        world.insert_component(
            entity,
            crate::ecs::component::MotionPath {
                center: cgmath::Vector3::new(mp.center[0], mp.center[1], mp.center[2]),
                radius: mp.radius,
                angular_speed: mp.angular_speed,
                phase_offset: mp.phase_offset,
                enabled: mp.enabled,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_channel_scene_name_roundtrip() {
        for domain in crate::ecs::component::scalar_channel_domains() {
            for channel in domain.channels {
                let (_, found) =
                    crate::ecs::component::scalar_channel_for_scene_name(channel.scene_name)
                        .expect("scene name resolves");
                assert_eq!(
                    found.code, channel.code,
                    "Roundtrip failed for {}",
                    channel.scene_name
                );
            }
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
                contour_wiggle_amp: 0.3,
                aniso_axis_advect: 0.0,
                rte_bands: 4.0,
                sigma_dispersion: 1.0,
                edge_temperature_blend: 0.0,
                tip_carve_depth: 1.0,
                tip_carve_reach: 0.2,
                warp_reach: default_warp_reach(),
            },
            channels: vec![FlameChannelData {
                param: "Height".to_string(),
                keys: vec![
                    FlameKeyData {
                        time: 0.0,
                        value: 1.0,
                        interpolation: "Linear".to_string(),
                        in_tangent: Some([0.0, 0.0]),
                        out_tangent: Some([0.0, 0.0]),
                        weight_mode: Some("NonWeighted".to_string()),
                    },
                    FlameKeyData {
                        time: 2.0,
                        value: 2.0,
                        interpolation: "Linear".to_string(),
                        in_tangent: Some([0.0, 0.0]),
                        out_tangent: Some([0.0, 0.0]),
                        weight_mode: Some("NonWeighted".to_string()),
                    },
                ],
            }],
            motion_path: None,
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
        assert_eq!(
            scene.effect.temperature_base_k,
            restored.effect.temperature_base_k
        );
        assert_eq!(
            scene.effect.temperature_tip_k,
            restored.effect.temperature_tip_k
        );
        assert_eq!(scene.effect.use_blackbody, restored.effect.use_blackbody);
        assert_eq!(
            scene.effect.noise_amplitude,
            restored.effect.noise_amplitude
        );
        assert_eq!(
            scene.effect.noise_frequency,
            restored.effect.noise_frequency
        );
        assert_eq!(
            scene.effect.noise_scroll_speed,
            restored.effect.noise_scroll_speed
        );
        assert_eq!(scene.effect.time_scale, restored.effect.time_scale);
        assert_eq!(scene.effect.time_offset, restored.effect.time_offset);
        assert_eq!(scene.effect.warp_amp, restored.effect.warp_amp);
        assert_eq!(scene.effect.warp_freq, restored.effect.warp_freq);
        assert_eq!(scene.effect.rise_speed, restored.effect.rise_speed);
        assert_eq!(scene.effect.taper_power, restored.effect.taper_power);
        assert_eq!(
            scene.effect.radius_tip_ratio,
            restored.effect.radius_tip_ratio
        );
        assert_eq!(scene.effect.edge_low, restored.effect.edge_low);
        assert_eq!(scene.effect.edge_high, restored.effect.edge_high);
        assert_eq!(scene.effect.white_boost, restored.effect.white_boost);
        assert_eq!(scene.effect.wind_direction, restored.effect.wind_direction);
        assert_eq!(scene.effect.bend_amount, restored.effect.bend_amount);
        assert_eq!(scene.effect.bend_power, restored.effect.bend_power);
        assert_eq!(
            scene.effect.self_shadow_strength,
            restored.effect.self_shadow_strength
        );
        assert_eq!(scene.channels.len(), restored.channels.len());
        assert_eq!(scene.channels[0].param, restored.channels[0].param);
        assert_eq!(
            scene.channels[0].keys.len(),
            restored.channels[0].keys.len()
        );
        for (k1, k2) in scene.channels[0]
            .keys
            .iter()
            .zip(restored.channels[0].keys.iter())
        {
            assert_eq!(k1.time, k2.time);
            assert_eq!(k1.value, k2.value);
            assert_eq!(k1.interpolation, k2.interpolation);
            assert_eq!(k1.in_tangent, k2.in_tangent);
            assert_eq!(k1.out_tangent, k2.out_tangent);
            assert_eq!(k1.weight_mode, k2.weight_mode);
        }
    }

    #[test]
    fn test_flame_key_data_bezier_roundtrip() {
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
                contour_wiggle_amp: 0.3,
                aniso_axis_advect: 0.0,
                rte_bands: 4.0,
                sigma_dispersion: 1.0,
                edge_temperature_blend: 0.0,
                tip_carve_depth: 1.0,
                tip_carve_reach: 0.2,
                warp_reach: default_warp_reach(),
            },
            channels: vec![FlameChannelData {
                param: "Height".to_string(),
                keys: vec![
                    FlameKeyData {
                        time: 0.0,
                        value: 1.0,
                        interpolation: "Bezier".to_string(),
                        in_tangent: Some([0.0, 0.0]),
                        out_tangent: Some([0.5, 0.3]),
                        weight_mode: Some("Weighted".to_string()),
                    },
                    FlameKeyData {
                        time: 2.0,
                        value: 3.0,
                        interpolation: "Bezier".to_string(),
                        in_tangent: Some([-0.4, -0.2]),
                        out_tangent: Some([0.0, 0.0]),
                        weight_mode: Some("NonWeighted".to_string()),
                    },
                ],
            }],
            motion_path: None,
        };

        let json = serde_json::to_string(&scene).expect("Failed to serialize");
        let restored: FlameSceneData = serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(scene.channels.len(), restored.channels.len());
        assert_eq!(
            scene.channels[0].keys.len(),
            restored.channels[0].keys.len()
        );
        for (k1, k2) in scene.channels[0]
            .keys
            .iter()
            .zip(restored.channels[0].keys.iter())
        {
            assert_eq!(k1.time, k2.time);
            assert_eq!(k1.value, k2.value);
            assert_eq!(k1.interpolation, k2.interpolation);
            assert_eq!(k1.in_tangent, k2.in_tangent);
            assert_eq!(k1.out_tangent, k2.out_tangent);
            assert_eq!(k1.weight_mode, k2.weight_mode);
        }
    }

    #[test]
    fn test_flame_key_data_old_format_load() {
        // JSON with no tangent fields (old format) — should load with None defaults
        let json = r#"{
            "effect": {
                "position": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "height": 1.0,
                "radius": 0.5,
                "sigma_t": 0.3,
                "intensity": 1.0,
                "color_base": [1.0, 0.5, 0.0],
                "color_tip": [1.0, 1.0, 1.0],
                "temperature_base_k": 3200.0,
                "temperature_tip_k": 1500.0,
                "use_blackbody": true,
                "noise_amplitude": 0.1,
                "noise_frequency": 1.0,
                "noise_scroll_speed": 0.0,
                "time_scale": 1.0,
                "time_offset": 0.0,
                "warp_amp": 0.05,
                "warp_freq": 2.0,
                "rise_speed": 1.0,
                "taper_power": 1.0,
                "radius_tip_ratio": 0.10,
                "edge_low": 0.3,
                "edge_high": 0.7,
                "white_boost": 0.0,
                "wind_direction": [0.0, 0.0],
                "bend_amount": 0.0,
                "bend_power": 2.0,
                "self_shadow_strength": 0.0,
                "envelope_peak": 0.35,
                "envelope_base": 0.45,
                "envelope_tail": 1.6,
                "radial_sharpness": 4.0,
                "occlusion_lum_ref": 1.0,
                "contour_wiggle_amp": 0.3
            },
            "channels": [
                {
                    "param": "Height",
                    "keys": [
                        {"time": 0.0, "value": 1.0},
                        {"time": 2.0, "value": 2.0}
                    ]
                }
            ]
        }"#;

        let scene: FlameSceneData =
            serde_json::from_str(json).expect("Failed to deserialize old format");

        assert_eq!(scene.channels.len(), 1);
        assert_eq!(scene.channels[0].keys.len(), 2);
        // Old format keys should have None for tangent fields and default interpolation
        for k in &scene.channels[0].keys {
            assert_eq!(k.in_tangent, None);
            assert_eq!(k.out_tangent, None);
            assert_eq!(k.weight_mode, None);
            assert_eq!(k.interpolation, "Linear");
        }
    }

    fn register_flame_clip_for_test(
        world: &mut crate::ecs::world::World,
        entity: crate::ecs::world::Entity,
        assets: &mut crate::asset::AssetStorage,
        clip: thyllore_anim_core::editable::EditableAnimationClip,
    ) -> thyllore_anim_core::editable::SourceClipId {
        world.insert_resource(crate::ecs::resource::ClipLibrary::new());
        let clip_id = {
            let mut lib = world.resource_mut::<crate::ecs::resource::ClipLibrary>();
            crate::ecs::systems::clip_library_systems::clip_library_register_and_activate(
                &mut lib, assets, clip,
            )
        };
        let mut schedule = crate::ecs::component::ClipSchedule::new();
        let instance_id = schedule.next_instance_id;
        schedule.next_instance_id += 1;
        schedule
            .instances
            .push(thyllore_anim_core::editable::ClipInstance::new(
                instance_id,
                clip_id,
                0.0,
            ));
        world.insert_component(entity, schedule);
        clip_id
    }

    #[test]
    fn test_flame_clip_world_roundtrip_bezier() {
        use thyllore_anim_core::editable::{InterpolationType, TangentWeightMode};

        // Build a world with a flame entity whose scheduled clip has Bezier scalar keys
        let mut world = crate::ecs::world::World::new();
        let entity = crate::ecs::systems::spawn_flame(
            &mut world,
            crate::ecs::systems::DEFAULT_FLAME_NAME,
            crate::ecs::component::FlameEffect::default(),
        );

        let mut clip = thyllore_anim_core::editable::EditableAnimationClip::new(
            0,
            crate::ecs::component::FLAME_DOMAIN.name.to_string(),
        );
        {
            let curve = clip
                .get_or_add_scalar_curve(crate::ecs::component::FlameParam::Height.property_type());
            let id0 = thyllore_anim_core::editable::curve_add_keyframe(curve, 0.0, 1.0);
            let id1 = thyllore_anim_core::editable::curve_add_keyframe(curve, 2.0, 3.0);
            for (id, wm) in [
                (id0, TangentWeightMode::Weighted),
                (id1, TangentWeightMode::NonWeighted),
            ] {
                let key = curve.get_keyframe_mut(id).unwrap();
                key.interpolation = InterpolationType::Bezier;
                key.in_tangent.time_offset = -0.4;
                key.in_tangent.value_offset = -0.2;
                key.out_tangent.time_offset = 0.5;
                key.out_tangent.value_offset = 0.3;
                key.weight_mode = wm;
            }
        }
        let mut assets = crate::asset::AssetStorage::new();
        register_flame_clip_for_test(&mut world, entity, &mut assets, clip);

        // Save -> apply to a fresh world -> compare
        let data = build_flame_scene_data(&world).expect("scene data");
        assert_eq!(data.channels.len(), 1);
        assert_eq!(data.channels[0].param, "Height");

        let mut world2 = crate::ecs::world::World::new();
        let entity2 = crate::ecs::systems::spawn_flame(
            &mut world2,
            crate::ecs::systems::DEFAULT_FLAME_NAME,
            crate::ecs::component::FlameEffect::default(),
        );
        world2.insert_resource(crate::ecs::resource::ClipLibrary::new());
        let mut assets2 = crate::asset::AssetStorage::new();
        apply_flame_state_to_world(&mut world2, &mut assets2, &data);

        let clip_id2 =
            crate::ecs::systems::find_entity_clip_id(&world2, entity2).expect("clip scheduled");
        let lib2 = world2
            .get_resource::<crate::ecs::resource::ClipLibrary>()
            .unwrap();
        let clip2 = lib2.get(clip_id2).expect("clip registered");
        let curve2 = clip2
            .get_scalar_curve(crate::ecs::component::FlameParam::Height.property_type())
            .expect("scalar curve restored");
        assert_eq!(curve2.keyframes.len(), 2);
        let k0 = &curve2.keyframes[0];
        assert_eq!(k0.time, 0.0);
        assert_eq!(k0.value, 1.0);
        assert_eq!(k0.interpolation, InterpolationType::Bezier);
        assert_eq!(k0.in_tangent.time_offset, -0.4);
        assert_eq!(k0.out_tangent.value_offset, 0.3);
        assert_eq!(k0.weight_mode, TangentWeightMode::Weighted);
        let k1 = &curve2.keyframes[1];
        assert_eq!(k1.time, 2.0);
        assert_eq!(k1.weight_mode, TangentWeightMode::NonWeighted);
    }

    #[test]
    fn test_motion_path_world_roundtrip() {
        // Build a world with a flame entity + MotionPath with non-trivial values
        let mut world = crate::ecs::world::World::new();
        let entity = crate::ecs::systems::spawn_flame(
            &mut world,
            crate::ecs::systems::DEFAULT_FLAME_NAME,
            crate::ecs::component::FlameEffect::default(),
        );
        world.insert_component(
            entity,
            crate::ecs::component::MotionPath {
                center: cgmath::Vector3::new(1.0, 2.0, 3.0),
                radius: 5.0,
                angular_speed: 0.7,
                phase_offset: 1.5,
                enabled: true,
            },
        );

        // Save: build FlameSceneData from world
        let scene_data = build_flame_scene_data(&world).expect("build_flame_scene_data failed");

        // Serialize to JSON and back (simulating file roundtrip)
        let json = serde_json::to_string(&scene_data).expect("serialize failed");
        let restored: FlameSceneData = serde_json::from_str(&json).expect("deserialize failed");

        // Verify motion_path is Some with correct values
        let mp = restored
            .motion_path
            .as_ref()
            .expect("motion_path should be Some");
        assert_eq!(mp.center, [1.0, 2.0, 3.0]);
        assert_eq!(mp.radius, 5.0);
        assert_eq!(mp.angular_speed, 0.7);
        assert_eq!(mp.phase_offset, 1.5);
        assert!(mp.enabled);

        // Load: apply back to a fresh world
        let mut world2 = crate::ecs::world::World::new();
        let entity2 = crate::ecs::systems::spawn_flame(
            &mut world2,
            crate::ecs::systems::DEFAULT_FLAME_NAME,
            crate::ecs::component::FlameEffect::default(),
        );
        world2.insert_resource(crate::ecs::resource::ClipLibrary::new());
        let mut assets2 = crate::asset::AssetStorage::new();
        apply_flame_state_to_world(&mut world2, &mut assets2, &restored);

        // Verify MotionPath component was inserted with correct values
        let loaded_mp = world2
            .get_component::<crate::ecs::component::MotionPath>(entity2)
            .expect("MotionPath should be inserted");
        assert_eq!(loaded_mp.center.x, 1.0);
        assert_eq!(loaded_mp.center.y, 2.0);
        assert_eq!(loaded_mp.center.z, 3.0);
        assert_eq!(loaded_mp.radius, 5.0);
        assert_eq!(loaded_mp.angular_speed, 0.7);
        assert_eq!(loaded_mp.phase_offset, 1.5);
        assert!(loaded_mp.enabled);
    }

    #[test]
    fn test_old_json_without_motion_path_loads() {
        // Old JSON format without motion_path field should deserialize with motion_path = None
        let json = r#"{
            "effect": {
                "position": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "height": 1.0,
                "radius": 0.5,
                "sigma_t": 0.3,
                "intensity": 1.0,
                "color_base": [1.0, 0.5, 0.0],
                "color_tip": [1.0, 1.0, 1.0],
                "temperature_base_k": 3200.0,
                "temperature_tip_k": 1500.0,
                "use_blackbody": true,
                "noise_amplitude": 0.1,
                "noise_frequency": 1.0,
                "noise_scroll_speed": 0.0,
                "time_scale": 1.0,
                "time_offset": 0.0,
                "warp_amp": 0.05,
                "warp_freq": 2.0,
                "rise_speed": 1.0,
                "taper_power": 1.0,
                "radius_tip_ratio": 0.10,
                "edge_low": 0.3,
                "edge_high": 0.7,
                "white_boost": 0.0,
                "wind_direction": [0.0, 0.0],
                "bend_amount": 0.0,
                "bend_power": 2.0,
                "self_shadow_strength": 0.0,
                "envelope_peak": 0.35,
                "envelope_base": 0.45,
                "envelope_tail": 1.6,
                "radial_sharpness": 4.0,
                "occlusion_lum_ref": 1.0,
                "contour_wiggle_amp": 0.3
            },
            "channels": []
        }"#;

        let scene: FlameSceneData = serde_json::from_str(json)
            .expect("Failed to deserialize old format without motion_path");

        assert!(
            scene.motion_path.is_none(),
            "motion_path should be None for old format"
        );
    }
}
