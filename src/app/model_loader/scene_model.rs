use cgmath::SquareMatrix;

use crate::animation::editable::SourceClipId;
use crate::asset::{AssetStorage, MeshAsset, NodeAsset, SkeletonAsset};
use crate::ecs::component::{AnimationMeta, ClipSchedule, EntityIcon};
use crate::ecs::resource::gizmo::{BoneGizmoData, ConstraintGizmoData};
use crate::ecs::resource::{
    AnimationType, BatchRun, ClipLibrary, FbxModelCache, GltfModelCache, ModelState, NodeAssets,
    TimelineState,
};
use crate::ecs::world::{Animator, Transform, World};
use crate::loader::fbx::FbxModel;
use crate::loader::ModelLoadResult;
use crate::vulkanr::resource::graphics_resource::{GraphicsResources, NodeData};

pub(super) fn insert_model_caches(
    world: &mut World,
    model_name: &str,
    fbx_model: Option<FbxModel>,
) {
    if let Some(fbx) = fbx_model {
        let needs_coord_conversion = fbx.fbx_data.iter().any(|d| !d.clusters.is_empty());
        world.insert_resource(FbxModelCache::new(
            fbx,
            model_name.to_string(),
            needs_coord_conversion,
        ));
        world.insert_resource(GltfModelCache::empty());
    } else {
        world.insert_resource(FbxModelCache::empty());
        let path_lower = model_name.to_lowercase();
        if path_lower.ends_with(".gltf") || path_lower.ends_with(".glb") {
            world.insert_resource(GltfModelCache::new(model_name.to_string()));
        } else {
            world.insert_resource(GltfModelCache::empty());
        }
    }
}

pub(super) fn determine_animation_type(load_result: &ModelLoadResult) -> AnimationType {
    if load_result.has_skinned_meshes {
        AnimationType::Skeletal
    } else if !load_result.clips.is_empty() {
        AnimationType::Node
    } else {
        AnimationType::None
    }
}

pub(super) fn log_model_load_info(
    load_result: &ModelLoadResult,
    animation_type: AnimationType,
    node_animation_scale: f32,
) {
    let mesh_category = if load_result.has_skinned_meshes {
        MeshCategory::Skinned
    } else {
        MeshCategory::Unskinned
    };
    let mesh_scale_debug = compute_bone_gizmo_mesh_scale(node_animation_scale, mesh_category);
    log!(
        "[ModelLoad] type={:?}, has_skinned={}, node_anim_scale={}, mesh_scale={}",
        animation_type,
        load_result.has_skinned_meshes,
        node_animation_scale,
        mesh_scale_debug
    );
}

pub fn find_best_clip(world: &World) -> Option<SourceClipId> {
    if !world.contains_resource::<ClipLibrary>() {
        return None;
    }

    let clip_library = world.resource::<ClipLibrary>();
    let mut clip_ids: Vec<_> = clip_library.all_clip_ids().copied().collect();
    clip_ids.sort_unstable();

    clip_ids
        .iter()
        .copied()
        .find(|&id| {
            clip_library
                .get(id)
                .is_some_and(|clip| !clip.tracks.is_empty())
        })
        .or_else(|| clip_ids.first().copied())
}

pub(super) fn restore_batch_playback(world: &World) {
    let requested_clip_id = match world.get_resource::<BatchRun>() {
        Some(batch_run) if batch_run.play_requested => batch_run.play_clip_id,
        _ => return,
    };

    let clip_still_loaded = requested_clip_id.is_some_and(|id| {
        world
            .get_resource::<ClipLibrary>()
            .is_some_and(|library| library.get(id).is_some())
    });
    let clip_id = if clip_still_loaded {
        requested_clip_id
    } else {
        find_best_clip(world)
    };

    if !world.contains_resource::<TimelineState>() {
        return;
    }

    let mut timeline = world.resource_mut::<TimelineState>();
    timeline.playing = true;
    if let Some(id) = clip_id {
        timeline.current_clip_id = Some(id);
    }
}

pub(super) fn setup_animation_system(
    world: &mut World,
    load_result: &ModelLoadResult,
    assets: &mut AssetStorage,
) {
    if world.contains_resource::<ClipLibrary>() {
        let mut clip_library = world.resource_mut::<ClipLibrary>();
        clip_library.animation = load_result.animation_system.clone();
        clip_library.morph_animation = load_result.morph_animation.clone();
    }

    for skeleton in &load_result.skeletons {
        let skeleton_asset = SkeletonAsset {
            id: 0,
            skeleton_id: skeleton.id,
            skeleton: skeleton.clone(),
        };
        assets.add_skeleton(skeleton_asset);
    }

    if world.contains_resource::<ModelState>() {
        let mut model_state = world.resource_mut::<ModelState>();
        model_state.has_skinned_meshes = load_result.has_skinned_meshes;
    }
}

pub(super) fn setup_nodes(world: &mut World, load_result: &ModelLoadResult) {
    let nodes: Vec<NodeData> = load_result
        .nodes
        .iter()
        .map(|n| NodeData {
            index: n.index,
            name: n.name.clone(),
            parent_index: n.parent_index,
            local_transform: n.local_transform,
            global_transform: cgmath::Matrix4::identity(),
        })
        .collect();

    let node_count = nodes.len();

    if world.contains_resource::<NodeAssets>() {
        let mut node_assets = world.resource_mut::<NodeAssets>();
        node_assets.nodes = nodes;
    }

    log!("Loaded {} nodes into NodeAssets", node_count);
}

pub(super) fn create_ecs_entities(
    model_name: &str,
    graphics: &GraphicsResources,
    world: &mut World,
    assets: &mut AssetStorage,
    animation_type: AnimationType,
    node_animation_scale: f32,
    loaded_clips: &[crate::animation::AnimationClip],
    scene_will_provide_clips: bool,
) -> crate::ecs::world::Entity {
    let name = std::path::Path::new(model_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();

    ensure_ecs_resources(world);

    let mut first_editable_clip_id =
        register_clips_to_library(world, assets, loaded_clips, scene_will_provide_clips);

    register_node_assets(world, assets);

    if first_editable_clip_id.is_none() && !scene_will_provide_clips && !assets.skeletons.is_empty()
    {
        first_editable_clip_id = Some(register_empty_editable_clip(world, assets));
    }

    let has_playable_clip = first_editable_clip_id.is_some();

    let initial_schedule = if has_playable_clip && !scene_will_provide_clips {
        build_initial_clip_schedule(first_editable_clip_id, world)
    } else {
        ClipSchedule::new()
    };

    let mut parent_builder = world
        .entity()
        .with_name(&name)
        .with_transform(Transform::default())
        .with_visible(true)
        .with_editor_display(EntityIcon::Model, true);

    if has_playable_clip {
        parent_builder = parent_builder
            .with_animator(Animator::new())
            .with_clip_schedule(initial_schedule)
            .with_animation_meta(AnimationMeta {
                animation_type,
                node_animation_scale,
            });
    }

    let parent_entity = parent_builder.build();

    log!(
        "Created parent entity '{}': entity_id={}",
        name,
        parent_entity
    );

    build_mesh_entities(&name, graphics, world, assets, parent_entity);

    log!(
        "Created {} ECS entities, {} mesh assets, {} skeletons, {} clips, {} nodes",
        world.entity_count(),
        assets.meshes.len(),
        assets.skeletons.len(),
        assets.animation_clips.len(),
        assets.nodes.len()
    );

    parent_entity
}

pub(super) fn ensure_ecs_resources(world: &mut World) {
    if !world.contains_resource::<ClipLibrary>() {
        world.insert_resource(ClipLibrary::new());
    }
    if !world.contains_resource::<TimelineState>() {
        world.insert_resource(TimelineState::new());
    }
    if !world.contains_resource::<crate::ecs::resource::KeyframeCopyBuffer>() {
        world.insert_resource(crate::ecs::resource::KeyframeCopyBuffer::default());
    }
    if !world.contains_resource::<crate::ecs::resource::EditHistory>() {
        world.insert_resource(crate::ecs::resource::EditHistory::new(100));
    }
    if !world.contains_resource::<crate::ecs::resource::ClipBrowserState>() {
        world.insert_resource(crate::ecs::resource::ClipBrowserState::default());
    }
    if !world.contains_resource::<crate::ecs::resource::BonePoseOverride>() {
        world.insert_resource(crate::ecs::resource::BonePoseOverride::default());
    }
}

fn register_clips_to_library(
    world: &mut World,
    assets: &mut AssetStorage,
    loaded_clips: &[crate::animation::AnimationClip],
    scene_will_provide_clips: bool,
) -> Option<SourceClipId> {
    if scene_will_provide_clips {
        return None;
    }

    let bone_names: std::collections::HashMap<u32, String> = assets
        .skeletons
        .values()
        .flat_map(|sa| sa.skeleton.bones.iter().map(|b| (b.id, b.name.clone())))
        .collect();

    let mut first_editable_clip_id = None;
    let mut clip_library = world.resource_mut::<ClipLibrary>();

    for clip in loaded_clips {
        let editable_id =
            crate::ecs::systems::clip_library_systems::clip_library_create_from_imported(
                &mut clip_library,
                assets,
                clip,
                &bone_names,
            );
        if first_editable_clip_id.is_none() {
            first_editable_clip_id = Some(editable_id);
        }
        log!(
            "Registered clip '{}' (source_id={})",
            clip.name,
            editable_id,
        );
    }
    drop(clip_library);

    if let Some(editable_id) = first_editable_clip_id {
        let clip_duration = world
            .resource::<ClipLibrary>()
            .get(editable_id)
            .map(|c| c.duration)
            .unwrap_or(0.0);
        let mut timeline_state = world.resource_mut::<TimelineState>();
        timeline_state.current_clip_id = Some(editable_id);
        crate::ecs::systems::timeline_apply_fit_zoom(&mut timeline_state, clip_duration);
        log!("Set timeline current_clip_id to {}", editable_id);
    }

    first_editable_clip_id
}

const EMPTY_CLIP_DEFAULT_DURATION_SECONDS: f32 = 5.0;

fn register_empty_editable_clip(world: &mut World, assets: &mut AssetStorage) -> SourceClipId {
    use crate::animation::editable::EditableAnimationClip;

    let mut clip_library = world.resource_mut::<ClipLibrary>();
    let mut editable = EditableAnimationClip::new(0, "New Animation".to_string());
    editable.duration = EMPTY_CLIP_DEFAULT_DURATION_SECONDS;
    let source_id = crate::ecs::systems::clip_library_systems::clip_library_register_and_activate(
        &mut clip_library,
        assets,
        editable,
    );
    drop(clip_library);

    let mut timeline_state = world.resource_mut::<TimelineState>();
    timeline_state.current_clip_id = Some(source_id);
    drop(timeline_state);

    log!(
        "Auto-created empty animation clip 'New Animation' (source_id={}, duration={}s) for model with no animations",
        source_id,
        EMPTY_CLIP_DEFAULT_DURATION_SECONDS
    );

    source_id
}

fn register_node_assets(world: &World, assets: &mut AssetStorage) {
    let node_assets = world.resource::<NodeAssets>();
    for node in &node_assets.nodes {
        let node_asset = NodeAsset {
            id: node.index as u64,
            name: node.name.clone(),
            parent_id: node.parent_index.map(|i| i as u64),
            local_transform: node.local_transform,
        };
        assets.add_node(node_asset);
    }
}

fn build_mesh_entities(
    name: &str,
    graphics: &GraphicsResources,
    world: &mut World,
    assets: &mut AssetStorage,
    parent_entity: crate::ecs::Entity,
) {
    for (mesh_idx, mesh) in graphics.meshes.iter().enumerate() {
        let entity_name = format!("{}_{:02}", name, mesh_idx + 1);

        let mesh_asset = MeshAsset {
            id: 0,
            name: entity_name.clone(),
            graphics_mesh_index: mesh_idx,
            object_index: mesh.object_index,
            material_id: graphics.mesh_material_ids.get(mesh_idx).copied(),
            skeleton_id: mesh.skeleton_id,
            node_index: mesh.node_index,
            render_to_gbuffer: mesh.render_to_gbuffer,
        };
        let asset_id = assets.add_mesh(mesh_asset);

        let entity = world
            .entity()
            .with_name(&entity_name)
            .with_global_transform()
            .with_visible(true)
            .with_parent(parent_entity)
            .with_editor_display(EntityIcon::Mesh, false)
            .with_mesh(asset_id, mesh.object_index)
            .build();

        log!(
            "Created ECS entity {} (asset_id={}) for mesh {}: entity_id={}, parent={}",
            entity_name,
            asset_id,
            mesh_idx,
            entity,
            parent_entity
        );
    }
}

pub(super) fn initialize_bone_gizmo_visibility(
    world: &mut World,
    assets: &AssetStorage,
    graphics: &GraphicsResources,
    node_animation_scale: f32,
    has_skinned_meshes: bool,
) {
    if !world.contains_resource::<BoneGizmoData>() {
        return;
    }

    let has_skeleton = !assets.skeletons.is_empty();
    let mut bone_gizmo = world.resource_mut::<BoneGizmoData>();

    bone_gizmo.stick_mesh.vertices.clear();
    bone_gizmo.stick_mesh.indices.clear();
    bone_gizmo.solid_mesh.vertices.clear();
    bone_gizmo.solid_mesh.indices.clear();
    bone_gizmo.wire_mesh.vertices.clear();
    bone_gizmo.wire_mesh.indices.clear();

    if has_skeleton {
        bone_gizmo.visible = true;

        let first_skeleton = assets.skeletons.values().next();
        if let Some(skel_asset) = first_skeleton {
            bone_gizmo.cached_skeleton_id = Some(skel_asset.skeleton_id);
            let category = if has_skinned_meshes {
                MeshCategory::Skinned
            } else {
                MeshCategory::Unskinned
            };
            bone_gizmo.mesh_scale = compute_bone_gizmo_mesh_scale(node_animation_scale, category);

            let skeleton = &skel_asset.skeleton;
            let rest_globals = crate::ecs::compute_pose_global_transforms(
                skeleton,
                &crate::ecs::create_pose_from_rest(skeleton),
            );

            bone_gizmo.bone_local_offsets =
                crate::ecs::compute_bone_local_offsets(skeleton, &rest_globals);
            bone_gizmo.cached_global_transforms = rest_globals;
        }
    } else {
        bone_gizmo.visible = false;
        bone_gizmo.cached_skeleton_id = None;
        bone_gizmo.cached_global_transforms.clear();
        bone_gizmo.bone_local_offsets.clear();
    }
}

enum MeshCategory {
    Skinned,
    Unskinned,
}

fn compute_bone_gizmo_mesh_scale(node_animation_scale: f32, category: MeshCategory) -> f32 {
    match category {
        MeshCategory::Skinned => 1.0,
        MeshCategory::Unskinned => node_animation_scale,
    }
}

pub(super) fn apply_loaded_constraints(load_result: &ModelLoadResult, world: &mut World) {
    use crate::ecs::component::{Constrained, ConstraintSet};

    if load_result.constraints.is_empty() {
        return;
    }

    let animator_entities = world.component_entities::<Animator>();
    if animator_entities.is_empty() {
        return;
    }

    let model_entity = animator_entities[0];

    let mut constraint_set = ConstraintSet::new();
    for loaded in &load_result.constraints {
        crate::ecs::systems::constraint_set_systems::constraint_set_add(
            &mut constraint_set,
            loaded.constraint_type.clone(),
            loaded.priority,
        );
    }

    world.insert_component(model_entity, constraint_set);
    world.insert_component(model_entity, Constrained);

    log!(
        "Applied {} constraints to entity {}",
        load_result.constraints.len(),
        model_entity
    );
}

pub(super) fn apply_loaded_spring_bones(load_result: &ModelLoadResult, world: &mut World) {
    use crate::ecs::component::{SpringBoneSetup, WithSpringBone};
    use crate::ecs::resource::SpringBoneState;

    let Some(ref setup) = load_result.spring_bone_setup else {
        return;
    };

    let animator_entities = world.component_entities::<Animator>();
    if animator_entities.is_empty() {
        return;
    }

    let model_entity = animator_entities[0];
    world.insert_component(model_entity, setup.clone());
    world.insert_component(model_entity, WithSpringBone);

    if !world.contains_resource::<SpringBoneState>() {
        world.insert_resource(SpringBoneState::default());
    }

    log!(
        "Applied VRMC spring bones to entity {}: {} chains, {} colliders",
        model_entity,
        setup.chains.len(),
        setup.colliders.len()
    );
}

pub(super) fn initialize_constraint_gizmo_visibility(world: &mut World) {
    if !world.contains_resource::<ConstraintGizmoData>() {
        return;
    }

    let has_bone_gizmo_visible = world
        .get_resource::<BoneGizmoData>()
        .map(|bg| bg.visible)
        .unwrap_or(false);

    let has_constraints = world.iter_constrained_entities().next().is_some();

    let mut cg = world.resource_mut::<ConstraintGizmoData>();
    cg.visible = has_bone_gizmo_visible && has_constraints;
}

pub fn build_initial_clip_schedule(
    first_source_id: Option<SourceClipId>,
    world: &World,
) -> ClipSchedule {
    let mut schedule = ClipSchedule::new();

    let Some(source_id) = first_source_id else {
        return schedule;
    };

    let clip_library = world.resource::<ClipLibrary>();
    let duration = clip_library
        .get(source_id)
        .map(|c| c.duration)
        .unwrap_or(1.0);
    drop(clip_library);

    crate::ecs::systems::clip_schedule_systems::clip_schedule_add_instance(
        &mut schedule,
        source_id,
        duration,
    );
    schedule
}

pub(super) fn build_mesh_entities_range(
    name: &str,
    graphics: &GraphicsResources,
    world: &mut World,
    assets: &mut AssetStorage,
    parent_entity: crate::ecs::Entity,
    mesh_range: std::ops::Range<usize>,
) {
    for mesh_idx in mesh_range {
        let mesh = &graphics.meshes[mesh_idx];
        let entity_name = format!("{}_{:02}", name, mesh_idx + 1);

        let mesh_asset = MeshAsset {
            id: 0,
            name: entity_name.clone(),
            graphics_mesh_index: mesh_idx,
            object_index: mesh.object_index,
            material_id: graphics.mesh_material_ids.get(mesh_idx).copied(),
            skeleton_id: mesh.skeleton_id,
            node_index: mesh.node_index,
            render_to_gbuffer: mesh.render_to_gbuffer,
        };
        let asset_id = assets.add_mesh(mesh_asset);

        let entity = world
            .entity()
            .with_name(&entity_name)
            .with_global_transform()
            .with_visible(true)
            .with_parent(parent_entity)
            .with_editor_display(EntityIcon::Mesh, false)
            .with_mesh(asset_id, mesh.object_index)
            .build();

        log!(
            "Created additive mesh entity {} (asset_id={}) for mesh {}: entity_id={}, parent={}",
            entity_name,
            asset_id,
            mesh_idx,
            entity,
            parent_entity
        );
    }
}
