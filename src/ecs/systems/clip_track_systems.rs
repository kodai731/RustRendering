use crate::animation::editable::SourceClipId;
use crate::animation::{BoneId, Skeleton};
use crate::asset::storage::AssetStorage;
use crate::ecs::component::{
    ClipGroupSnapshot, ClipInstanceSnapshot, ClipSchedule, ClipTrackEntry, ClipTrackSnapshot,
};
use crate::ecs::resource::{ClipLibrary, NodeAssets};
use crate::ecs::world::{Entity, MeshRef, Name, World};

pub fn query_clip_tracks(
    world: &World,
    clip_library: &ClipLibrary,
    assets: &AssetStorage,
) -> ClipTrackSnapshot {
    let mut entries = Vec::new();

    for (entity, schedule) in world.iter_components::<ClipSchedule>() {
        let entity_name = world
            .get_component::<Name>(entity)
            .map(|n| n.0.clone())
            .unwrap_or_else(|| format!("Entity {}", entity));

        let first_source_id = schedule.instances.first().map(|i| i.source_id);
        let mesh_entity = world
            .find_child_mesh_entities(entity)
            .into_iter()
            .next()
            .unwrap_or(entity);
        let mesh_bone_id =
            resolve_mesh_bone_id(world, mesh_entity, assets, clip_library, first_source_id);

        let instances: Vec<ClipInstanceSnapshot> = schedule
            .instances
            .iter()
            .map(|inst| {
                let clip_name = clip_library
                    .get_source(inst.source_id)
                    .map(|s| s.name().to_string())
                    .unwrap_or_else(|| format!("Clip {}", inst.source_id));

                let group_id =
                    crate::ecs::systems::clip_schedule_systems::clip_schedule_find_group(
                        schedule,
                        inst.instance_id,
                    )
                    .map(|g| g.id);

                ClipInstanceSnapshot {
                    instance_id: inst.instance_id,
                    source_id: inst.source_id,
                    clip_name,
                    start_time: inst.start_time,
                    end_time: inst.end_time(),
                    clip_in: inst.clip_in,
                    clip_out: inst.clip_out,
                    muted: inst.muted,
                    weight: inst.weight,
                    blend_mode: inst.blend_mode,
                    group_id,
                }
            })
            .collect();

        let groups: Vec<ClipGroupSnapshot> = schedule
            .groups
            .iter()
            .map(|g| ClipGroupSnapshot {
                id: g.id,
                name: g.name.clone(),
                muted: g.muted,
                weight: g.weight,
                instance_ids: g.instance_ids.clone(),
            })
            .collect();

        if !instances.is_empty() {
            entries.push(ClipTrackEntry {
                entity,
                entity_name,
                mesh_bone_id,
                instances,
                groups,
            });
        }
    }

    ClipTrackSnapshot {
        entries,
        camera_switch_markers: crate::ecs::systems::camera_switch_markers(world),
    }
}

pub fn resolve_mesh_bone_id(
    world: &World,
    entity: Entity,
    assets: &AssetStorage,
    clip_library: &ClipLibrary,
    source_id: Option<SourceClipId>,
) -> Option<BoneId> {
    let mesh_ref = world.get_component::<MeshRef>(entity)?;
    let mesh_asset = assets.get_mesh(mesh_ref.mesh_asset_id)?;
    let skeleton_id = mesh_asset.skeleton_id?;
    let skeleton = assets.get_skeleton_by_skeleton_id(skeleton_id)?;
    let node_index = mesh_asset.node_index?;

    let node_name = find_node_name_by_index(world, node_index)?;
    let bone_id = *skeleton.bone_name_to_id.get(&node_name)?;

    let clip = source_id.and_then(|id| clip_library.get(id));
    match clip {
        Some(c) if c.tracks.contains_key(&bone_id) => Some(bone_id),
        Some(c) => find_ancestor_with_track(skeleton, bone_id, c),
        None => Some(bone_id),
    }
}

fn find_node_name_by_index(world: &World, node_index: usize) -> Option<String> {
    let node_assets = world.get_resource::<NodeAssets>()?;
    node_assets
        .nodes
        .iter()
        .find(|n| n.index == node_index)
        .map(|n| n.name.clone())
}

fn find_ancestor_with_track(
    skeleton: &Skeleton,
    bone_id: BoneId,
    clip: &crate::animation::editable::EditableAnimationClip,
) -> Option<BoneId> {
    let mut current = bone_id;
    for _ in 0..skeleton.bones.len() {
        let parent_id = skeleton.bones.get(current as usize)?.parent_id?;
        if clip.tracks.contains_key(&parent_id) {
            return Some(parent_id);
        }
        current = parent_id;
    }
    None
}
