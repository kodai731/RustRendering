use serde::{Deserialize, Serialize};

pub const SCENE_FORMAT_VERSION: u32 = 5;

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
    #[serde(default)]
    pub water: Option<WaterSceneData>,
    #[serde(default)]
    pub wind: Option<WindSceneData>,
    #[serde(default)]
    pub debug_primitives: Vec<DebugPrimitiveSceneData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugPrimitiveSceneData {
    pub kind: String,
    pub position: [f32; 3],
}

pub fn debug_primitive_kind_to_str(kind: crate::ecs::events::DebugPrimitiveKind) -> &'static str {
    match kind {
        crate::ecs::events::DebugPrimitiveKind::Cube => "cube",
        crate::ecs::events::DebugPrimitiveKind::Sphere => "sphere",
        crate::ecs::events::DebugPrimitiveKind::Floor => "floor",
    }
}

pub fn debug_primitive_kind_from_str(s: &str) -> Option<crate::ecs::events::DebugPrimitiveKind> {
    match s {
        "cube" => Some(crate::ecs::events::DebugPrimitiveKind::Cube),
        "sphere" => Some(crate::ecs::events::DebugPrimitiveKind::Sphere),
        "floor" => Some(crate::ecs::events::DebugPrimitiveKind::Floor),
        _ => None,
    }
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
            water: None,
            wind: None,
            debug_primitives: Vec::new(),
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
    pub transform: crate::ecs::world::Transform,
}

impl ModelReference {
    /// Written in place of a file path when the mesh was generated in-app.
    pub const GENERATED_MESH: &'static str = "Generated Mesh";

    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            transform: crate::ecs::world::Transform::default(),
        }
    }

    pub fn is_generated_mesh(&self) -> bool {
        self.path == Self::GENERATED_MESH
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
    pub effect: thyllore_effect_core::FlameEffect,
    #[serde(default)]
    pub channels: Vec<FlameChannelData>,
    /// Authored clip length floor in seconds (0 = keyframes decide).
    #[serde(default)]
    pub clip_min_duration: f32,
    #[serde(default)]
    pub motion_path: Option<crate::ecs::component::MotionPath>,
    #[serde(default)]
    pub style: Option<FlameStyleRefData>,
}

/// Name and version of the style whose values are baked into the effect —
/// provenance only, so editing the style file never changes a saved scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameStyleRefData {
    pub name: String,
    pub version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterSceneData {
    pub effect: thyllore_effect_core::WaterTorusEffect,
    #[serde(default)]
    pub channels: Vec<FlameChannelData>,
    /// Authored clip length floor in seconds (0 = keyframes decide).
    #[serde(default)]
    pub clip_min_duration: f32,
    #[serde(default)]
    pub preset: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindSceneData {
    pub effect: thyllore_effect_core::WindTornadoEffect,
    #[serde(default)]
    pub channels: Vec<FlameChannelData>,
    #[serde(default)]
    pub clip_min_duration: f32,
    #[serde(default)]
    pub preset: Option<String>,
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

    let channels: Vec<FlameChannelData> = build_effect_channels_from_clip(world, *entity);
    let clip_min_duration = effect_clip_min_duration(world, *entity);

    let motion_path = world
        .get_component::<crate::ecs::component::MotionPath>(*entity)
        .map(|mp| mp.clone());

    let style = world
        .get_component::<crate::ecs::component::AppliedFlameStyle>(*entity)
        .map(|applied| FlameStyleRefData {
            name: applied.name.clone(),
            version: applied.version,
        });

    Some(FlameSceneData {
        effect: effect.clone(),
        channels,
        clip_min_duration,
        motion_path,
        style,
    })
}

fn effect_clip_min_duration(
    world: &crate::ecs::world::World,
    entity: crate::ecs::world::Entity,
) -> f32 {
    crate::ecs::systems::find_entity_clip_id(world, entity)
        .and_then(|clip_id| {
            world
                .get_resource::<crate::ecs::resource::ClipLibrary>()
                .and_then(|lib| lib.get(clip_id).map(|clip| clip.min_duration))
        })
        .unwrap_or(0.0)
}

fn build_effect_channels_from_clip(
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
        None => crate::ecs::systems::spawn_flame_with_clip(
            world,
            assets,
            crate::ecs::systems::DEFAULT_FLAME_NAME,
            crate::ecs::component::FlameEffect::default(),
        ),
    };

    if let Some(mut effect) = world.get_component_mut::<crate::ecs::component::FlameEffect>(entity)
    {
        thyllore_effect_core::overwrite_persisted_fields(&mut effect, &flame.effect);
        thyllore_effect_core::refresh_flame_coefficients(&mut effect, &Default::default());
    }

    if let Some(style) = &flame.style {
        world.insert_component(
            entity,
            crate::ecs::component::AppliedFlameStyle {
                name: style.name.clone(),
                version: style.version,
            },
        );
    }

    crate::ecs::systems::write_flame_transform(
        world,
        entity,
        flame.effect.position,
        flame.effect.rotation,
    );

    rebuild_effect_clip(
        world,
        assets,
        entity,
        crate::ecs::component::FLAME_DOMAIN.name,
        &flame.channels,
        flame.clip_min_duration,
    );

    if let Some(mp) = &flame.motion_path {
        world.insert_component(entity, mp.clone());
    }
}

/// Build WaterSceneData from the first water entity's WaterTorusEffect.
pub fn build_water_scene_data(world: &crate::ecs::world::World) -> Option<WaterSceneData> {
    let entities: Vec<_> = world.query_waters();
    let entity = entities.first()?;

    let effect = world.get_component::<crate::ecs::component::WaterTorusEffect>(*entity)?;

    let channels: Vec<FlameChannelData> = build_effect_channels_from_clip(world, *entity);
    let clip_min_duration = effect_clip_min_duration(world, *entity);

    let preset = world
        .get_component::<crate::ecs::component::AppliedWaterPreset>(*entity)
        .map(|applied| applied.name.clone());

    Some(WaterSceneData {
        effect: effect.clone(),
        channels,
        clip_min_duration,
        preset,
    })
}

pub fn build_wind_scene_data(world: &crate::ecs::world::World) -> Option<WindSceneData> {
    let entities: Vec<_> = world.query_winds();
    let entity = entities.first()?;
    let effect = world.get_component::<crate::ecs::component::WindTornadoEffect>(*entity)?;

    let preset = world
        .get_component::<crate::ecs::component::AppliedWindPreset>(*entity)
        .map(|applied| applied.name.clone());

    Some(WindSceneData {
        effect: effect.clone(),
        channels: build_effect_channels_from_clip(world, *entity),
        clip_min_duration: effect_clip_min_duration(world, *entity),
        preset,
    })
}

pub fn apply_wind_state_to_world(
    world: &mut crate::ecs::world::World,
    assets: &mut crate::asset::AssetStorage,
    wind: &WindSceneData,
) {
    let entities: Vec<_> = world.query_winds();
    let entity = match entities.first() {
        Some(e) => *e,
        None => crate::ecs::systems::spawn_wind_with_clip(
            world,
            assets,
            crate::ecs::systems::DEFAULT_WIND_NAME,
            crate::ecs::component::WindTornadoEffect::default(),
        ),
    };

    if let Some(mut effect) =
        world.get_component_mut::<crate::ecs::component::WindTornadoEffect>(entity)
    {
        thyllore_effect_core::overwrite_wind_persisted_fields(&mut effect, &wind.effect);
    }
    if let Some(ref preset_name) = wind.preset {
        world.insert_component(
            entity,
            crate::ecs::component::AppliedWindPreset {
                name: preset_name.clone(),
            },
        );
    }
    crate::ecs::systems::write_wind_transform(
        world,
        entity,
        wind.effect.position,
        wind.effect.rotation,
    );

    rebuild_effect_clip(
        world,
        assets,
        entity,
        crate::ecs::component::WIND_DOMAIN.name,
        &wind.channels,
        wind.clip_min_duration,
    );
}

pub fn build_debug_primitives_scene_data(
    world: &crate::ecs::world::World,
) -> Vec<DebugPrimitiveSceneData> {
    let mut primitives: Vec<_> = world
        .iter_components::<crate::ecs::component::DebugPrimitiveTag>()
        .filter_map(|(entity, tag)| {
            let transform = world.get_component::<crate::ecs::world::Transform>(entity)?;
            Some((
                entity,
                DebugPrimitiveSceneData {
                    kind: debug_primitive_kind_to_str(tag.kind).to_string(),
                    position: [
                        transform.translation.x,
                        transform.translation.y,
                        transform.translation.z,
                    ],
                },
            ))
        })
        .collect();

    primitives.sort_by_key(|(entity, _)| *entity);
    primitives.into_iter().map(|(_, data)| data).collect()
}

/// Apply loaded water state to the first water entity in the world.
pub fn apply_water_state_to_world(
    world: &mut crate::ecs::world::World,
    assets: &mut crate::asset::AssetStorage,
    water: &WaterSceneData,
) {
    let entities: Vec<_> = world.query_waters();
    let entity = match entities.first() {
        Some(e) => *e,
        None => crate::ecs::systems::spawn_water_with_clip(
            world,
            assets,
            "Water",
            crate::ecs::component::WaterTorusEffect::default(),
        ),
    };

    if let Some(mut effect) =
        world.get_component_mut::<crate::ecs::component::WaterTorusEffect>(entity)
    {
        thyllore_effect_core::overwrite_water_persisted_fields(&mut effect, &water.effect);
    }

    if let Some(ref preset_name) = water.preset {
        world.insert_component(
            entity,
            crate::ecs::component::AppliedWaterPreset {
                name: preset_name.clone(),
            },
        );
    }

    crate::ecs::systems::write_water_transform(
        world,
        entity,
        water.effect.position,
        water.effect.rotation,
    );

    rebuild_effect_clip(
        world,
        assets,
        entity,
        crate::ecs::component::WATER_DOMAIN.name,
        &water.channels,
        water.clip_min_duration,
    );
}

/// Rebuild an effect clip (scalar curves) from scene channels and schedule it on the entity.
/// Loading is idempotent: any previously scheduled clip instance is replaced. An empty clip
/// is still scheduled so the Timeline shows a lane whose length can be edited before any key exists.
fn rebuild_effect_clip(
    world: &mut crate::ecs::world::World,
    assets: &mut crate::asset::AssetStorage,
    entity: crate::ecs::world::Entity,
    domain_name: &str,
    channels: &[FlameChannelData],
    clip_min_duration: f32,
) {
    let mut editable =
        thyllore_anim_core::editable::EditableAnimationClip::new(0, domain_name.to_string());
    for ch in channels {
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
    editable.min_duration = clip_min_duration.max(0.0);
    thyllore_anim_core::editable::clip_recalculate_duration(&mut editable);

    if let Some(previous_clip_id) =
        crate::ecs::systems::scalar_clip_systems::find_entity_clip_id(world, entity)
    {
        world
            .resource_mut::<crate::ecs::resource::ClipLibrary>()
            .remove(previous_clip_id);
    }
    world.remove_component::<crate::ecs::component::ClipSchedule>(entity);
    let duration = editable.duration;
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
            duration,
        ));
    world.insert_component(entity, schedule);
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

    fn sample_flame_effect() -> thyllore_effect_core::FlameEffect {
        let mut effect = thyllore_effect_core::FlameEffect::default();
        effect.height = 1.0;
        effect.radius = 0.5;
        effect.sigma_t = 0.3;
        effect.intensity = 1.0;
        effect.color.base = [1.0, 0.5, 0.0];
        effect.color.tip = [1.0, 1.0, 1.0];
        effect.noise.amplitude = 0.1;
        effect.warp.amp = 0.05;
        effect.warp.freq = 2.0;
        effect.edge.low = 0.3;
        effect.edge.high = 0.7;
        effect.wind.bend_power = 2.0;
        effect
    }

    #[test]
    fn test_flame_style_ref_scene_roundtrip() {
        let scene = FlameSceneData {
            effect: sample_flame_effect(),
            channels: vec![],
            clip_min_duration: 0.0,
            motion_path: None,
            style: Some(FlameStyleRefData {
                name: "pillar-ref".to_string(),
                version: 1,
            }),
        };
        let json = serde_json::to_string(&scene).expect("serialize");
        let restored: FlameSceneData = serde_json::from_str(&json).expect("deserialize");
        let style = restored.style.expect("style ref survives");
        assert_eq!(style.name, "pillar-ref");
        assert_eq!(style.version, 1);
    }

    #[test]
    fn test_flame_effect_serde_fields_match_parameter_ownership_table() {
        let value = serde_json::to_value(sample_flame_effect()).expect("serialize");
        let serde_fields: std::collections::BTreeSet<String> = value
            .as_object()
            .expect("FlameEffect serializes to a flat object")
            .keys()
            .cloned()
            .collect();
        let table_fields: std::collections::BTreeSet<String> =
            thyllore_effect_core::PARAMETER_OWNERSHIP
                .iter()
                .map(|(name, _)| name.to_string())
                .collect();
        assert_eq!(serde_fields, table_fields);
    }

    #[test]
    fn test_flame_scene_data_serde_roundtrip() {
        let scene = FlameSceneData {
            effect: sample_flame_effect(),
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
            clip_min_duration: 0.0,
            motion_path: None,
            style: None,
        };

        let json = serde_json::to_string(&scene).expect("Failed to serialize FlameSceneData");
        let restored: FlameSceneData =
            serde_json::from_str(&json).expect("Failed to deserialize FlameSceneData");

        assert_eq!(
            serde_json::to_value(&scene.effect).expect("scene effect value"),
            serde_json::to_value(&restored.effect).expect("restored effect value")
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
            effect: sample_flame_effect(),
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
            clip_min_duration: 0.0,
            motion_path: None,
            style: None,
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
        assert_eq!(mp.center, cgmath::Vector3::new(1.0, 2.0, 3.0));
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
    fn test_apply_flame_state_spawns_entity_when_world_has_none() {
        let mut source = crate::ecs::world::World::new();
        let entity = crate::ecs::systems::spawn_flame(
            &mut source,
            crate::ecs::systems::DEFAULT_FLAME_NAME,
            crate::ecs::component::FlameEffect {
                height: 8.0,
                radius: 1.0,
                edge: thyllore_effect_core::FlameEdge {
                    radius_tip_ratio: 1.0,
                    ..thyllore_effect_core::FlameEdge::default()
                },
                ..crate::ecs::component::FlameEffect::default()
            },
        );
        let _ = entity;
        let data = build_flame_scene_data(&source).expect("scene data");

        let mut world2 = crate::ecs::world::World::new();
        world2.insert_resource(crate::ecs::resource::ClipLibrary::new());
        let mut assets2 = crate::asset::AssetStorage::new();
        assert!(world2.query_flames().is_empty());
        apply_flame_state_to_world(&mut world2, &mut assets2, &data);

        let flames = world2.query_flames();
        assert_eq!(flames.len(), 1, "flame entity should be spawned on load");
        let effect = world2
            .get_component::<crate::ecs::component::FlameEffect>(flames[0])
            .expect("FlameEffect on spawned entity");
        assert_eq!(effect.height, 8.0);
        assert_eq!(effect.radius, 1.0);
        assert_eq!(effect.edge.radius_tip_ratio, 1.0);
    }

    #[test]
    fn test_apply_flame_state_schedules_empty_clip_and_replaces_previous() {
        let mut source = crate::ecs::world::World::new();
        crate::ecs::systems::spawn_flame(
            &mut source,
            crate::ecs::systems::DEFAULT_FLAME_NAME,
            crate::ecs::component::FlameEffect::default(),
        );
        let data = build_flame_scene_data(&source).expect("scene data");
        assert!(data.channels.is_empty());

        let mut world2 = crate::ecs::world::World::new();
        world2.insert_resource(crate::ecs::resource::ClipLibrary::new());
        let mut assets2 = crate::asset::AssetStorage::new();
        apply_flame_state_to_world(&mut world2, &mut assets2, &data);
        let flame = world2.query_flames()[0];
        let first_clip =
            crate::ecs::systems::scalar_clip_systems::find_entity_clip_id(&world2, flame)
                .expect("keyless flame still gets a scheduled clip");

        apply_flame_state_to_world(&mut world2, &mut assets2, &data);
        let second_clip =
            crate::ecs::systems::scalar_clip_systems::find_entity_clip_id(&world2, flame)
                .expect("clip after reload");
        let library = world2.resource::<crate::ecs::resource::ClipLibrary>();
        assert_ne!(first_clip, second_clip);
        assert!(
            library.get(first_clip).is_none(),
            "previous clip is dropped"
        );
        assert!(library.get(second_clip).is_some());
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

#[cfg(test)]
mod legacy_format_golden {
    /// Default-flame JSON captured from the deleted FlameEffectData mirror; serde output must match it.
    const LEGACY_DEFAULT_EFFECT_JSON: &str = r#"{
  "position": [
    0.0,
    0.0,
    0.0
  ],
  "rotation": [
    1.0,
    0.0,
    0.0,
    0.0
  ],
  "height": 1.6,
  "radius": 0.6,
  "sigma_t": 1.0,
  "intensity": 2.2,
  "color_base": [
    1.0,
    0.45,
    0.1
  ],
  "color_tip": [
    1.0,
    0.1,
    0.02
  ],
  "temperature_base_k": 3200.0,
  "temperature_tip_k": 1500.0,
  "use_blackbody": true,
  "noise_amplitude": 1.5,
  "noise_contrast": 1.0,
  "noise_frequency": 6.0,
  "noise_scroll_speed": 1.0,
  "time_scale": 1.0,
  "time_offset": 0.0,
  "warp_amp": 1.4,
  "warp_freq": 5.0,
  "rise_speed": 1.5,
  "taper_power": 1.4,
  "radius_tip_ratio": 0.1,
  "edge_low": 0.27,
  "edge_high": 0.33,
  "white_boost": 4.0,
  "wind_direction": [
    0.0,
    0.0
  ],
  "bend_amount": 0.0,
  "bend_power": 1.7,
  "self_shadow_strength": 0.5,
  "envelope_peak": 0.25,
  "envelope_base": 0.05,
  "envelope_tail": 1.25,
  "radial_sharpness": 4.0,
  "occlusion_lum_ref": 1.0,
  "contour_wiggle_amp": 0.3,
  "aniso_axis_advect": 0.0,
  "rte_bands": 4.0,
  "sigma_dispersion": 1.0,
  "tip_carve_depth": 1.0,
  "tip_carve_reach": 0.2,
  "warp_reach": 0.35,
  "swirl_gain": 0.0,
  "swirl_speed": 1.0,
  "spread_gain": 0.0,
  "support_margin": 1.0,
  "meander_amp": 0.0,
  "meander_frequency": 1.0,
  "mix_lo": 0.0,
  "mix_hi": 2.0,
  "mix_height_gain": 0.0,
  "mix_scale": 1.0,
  "mix_radial_gain": 0.0,
  "density_exp": 1.0,
  "temp_exp": 1.0,
  "wien_c_k": 12000.0,
  "wave_segments": 64,
  "noise_aniso_y": 0.35,
  "edge_outer_sharpen": 0.0,
  "noise_scale_mode": 0.0,
  "erosion_noise_gain": 1.0,
  "twist_gain": 0.0,
  "twist_speed": 0.0,
  "burnout_gain": 0.0,
  "noise_shaping_scale": 0.0,
  "optical_depth": 0.0,
  "branch_period": 0.0,
  "branch_life": 2.5,
  "branch_gain": 0.0,
  "branch_core_radius": 0.35,
  "branch_core_offset": 0.0,
  "branch_reach": 1.5,
  "branch_spread": 0.3,
  "branch_spawn_height": 0.35,
  "branch_spawn_range": 0.4,
  "branch_seed": 0
}"#;

    #[test]
    fn test_default_flame_effect_matches_legacy_mirror_json() {
        let mut world = crate::ecs::world::World::new();
        crate::ecs::systems::spawn_flame(
            &mut world,
            crate::ecs::systems::DEFAULT_FLAME_NAME,
            crate::ecs::component::FlameEffect::default(),
        );
        let scene = super::build_flame_scene_data(&world).expect("scene data");

        let golden: serde_json::Value =
            serde_json::from_str(LEGACY_DEFAULT_EFFECT_JSON).expect("golden json");
        let printed = serde_json::to_string(&scene.effect).expect("effect json");
        let effect_value: serde_json::Value =
            serde_json::from_str(&printed).expect("effect json parses");
        assert_eq!(effect_value, golden);
    }

    #[test]
    fn test_legacy_mirror_json_roundtrips_through_flame_effect() {
        let loaded: thyllore_effect_core::FlameEffect =
            serde_json::from_str(LEGACY_DEFAULT_EFFECT_JSON).expect("golden deserializes");
        let golden: serde_json::Value =
            serde_json::from_str(LEGACY_DEFAULT_EFFECT_JSON).expect("golden json");
        let printed = serde_json::to_string(&loaded).expect("loaded json");
        let loaded_value: serde_json::Value =
            serde_json::from_str(&printed).expect("loaded json parses");
        assert_eq!(loaded_value, golden);
    }
}
