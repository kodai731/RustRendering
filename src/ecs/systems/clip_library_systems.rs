use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use anyhow::{Context, Result};

use crate::animation::editable::{
    clip_from_animation, clip_remap_bone_ids, clip_to_animation, EditableAnimationClip, SourceClip,
    SourceClipId,
};
use crate::animation::{AnimationClip, BoneId};
use crate::asset::{AnimationClipAsset, AssetStorage};
use crate::ecs::resource::ClipLibrary;
use crate::scene::{AnimationClipFile, ANIMATION_FORMAT_VERSION};

pub fn clip_library_register_and_activate(
    lib: &mut ClipLibrary,
    assets: &mut AssetStorage,
    editable: EditableAnimationClip,
) -> SourceClipId {
    let source_id = lib.next_source_id;
    lib.next_source_id += 1;

    let mut clip = editable;
    clip.id = source_id;

    let playable = clip_to_animation(&clip);
    let asset_id = assets.add_animation_clip(AnimationClipAsset {
        id: 0,
        clip: playable,
    });

    let source = SourceClip::new(source_id, clip);
    lib.source_clips.insert(source_id, source);
    lib.source_to_asset_id.insert(source_id, asset_id);

    source_id
}

pub fn clip_library_create_from_imported(
    lib: &mut ClipLibrary,
    assets: &mut AssetStorage,
    clip: &AnimationClip,
    bone_names: &HashMap<BoneId, String>,
) -> SourceClipId {
    let editable = crate::animation::editable::clip_from_animation(0, clip, bone_names);
    clip_library_register_and_activate(lib, assets, editable)
}

pub fn clip_library_to_playable(lib: &ClipLibrary, id: SourceClipId) -> Option<AnimationClip> {
    lib.source_clips
        .get(&id)
        .map(|s| clip_to_animation(&s.editable_clip))
}

pub fn clip_library_sync_dirty(lib: &mut ClipLibrary, assets: &mut AssetStorage) {
    for source_id in lib.dirty_sources.drain() {
        let (editable, asset_id) = match (
            lib.source_clips.get(&source_id),
            lib.source_to_asset_id.get(&source_id),
        ) {
            (Some(s), Some(&aid)) => (&s.editable_clip, aid),
            _ => continue,
        };

        let playable = clip_to_animation(editable);
        if let Some(asset) = assets.animation_clips.get_mut(&asset_id) {
            asset.clip = playable;
        }
    }
}

pub fn clip_library_clip_names(lib: &ClipLibrary) -> Vec<(SourceClipId, String)> {
    lib.source_clips
        .iter()
        .map(|(&id, s)| (id, s.editable_clip.name.clone()))
        .collect()
}

pub fn clip_library_update_save_metadata(
    lib: &mut ClipLibrary,
    id: SourceClipId,
    name: String,
    path: &Path,
) {
    if let Some(clip) = lib.get_mut(id) {
        clip.name = name;
        clip.source_path = Some(path.to_string_lossy().to_string());
    }
}

pub fn clip_library_save_to_file(lib: &ClipLibrary, id: SourceClipId, path: &Path) -> Result<()> {
    let source = lib.source_clips.get(&id).context("Clip not found")?;

    thyllore_exporter_core::systems::ron::export_ron_clip(&source.editable_clip, path)?;

    log!(
        "Saved animation clip '{}' to {:?}",
        source.editable_clip.name,
        path
    );
    Ok(())
}

pub fn clip_library_load_from_file(
    lib: &mut ClipLibrary,
    assets: &mut AssetStorage,
    path: &Path,
    bone_name_to_id: Option<&HashMap<String, BoneId>>,
) -> Result<SourceClipId> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Failed to read file: {:?}", path))?;

    let mut clip = deserialize_clip(&content)
        .with_context(|| format!("Failed to deserialize clip from: {:?}", path))?;

    if let Some(name_to_id) = bone_name_to_id {
        let needs_remap = clip.tracks.values().any(|track| {
            name_to_id
                .get(&track.bone_name)
                .map_or(false, |&expected_id| expected_id != track.bone_id)
        });

        if needs_remap {
            let remapped_count = clip.tracks.len();
            clip_remap_bone_ids(&mut clip, name_to_id);
            log!(
                "Remapped {} bone_ids by bone_name for '{}'",
                remapped_count,
                clip.name,
            );
        }
    }

    clip.source_path = Some(path.to_string_lossy().to_string());

    msg_info!("Loaded animation clip '{}' from {:?}", clip.name, path);

    let id = clip_library_register_and_activate(lib, assets, clip);
    Ok(id)
}

pub fn find_clip_schedule_owner(
    world: &crate::ecs::world::World,
    entity: crate::ecs::world::Entity,
) -> Option<crate::ecs::world::Entity> {
    use crate::ecs::component::ClipSchedule;
    use crate::ecs::world::Parent;

    let mut current = entity;
    loop {
        if world.get_component::<ClipSchedule>(current).is_some() {
            return Some(current);
        }
        match world.get_component::<Parent>(current) {
            Some(parent) => current = parent.0,
            None => return None,
        }
    }
}

pub fn query_selected_model_clip_ids(world: &crate::ecs::world::World) -> Vec<SourceClipId> {
    use crate::ecs::component::ClipSchedule;
    use crate::ecs::resource::HierarchyState;

    let Some(selected) = world.resource::<HierarchyState>().selected_entity else {
        return Vec::new();
    };
    let Some(owner) = find_clip_schedule_owner(world, selected) else {
        return Vec::new();
    };
    let Some(schedule) = world.get_component::<ClipSchedule>(owner) else {
        return Vec::new();
    };

    let mut clip_ids: Vec<SourceClipId> = schedule
        .instances
        .iter()
        .map(|inst| inst.source_id)
        .collect();
    clip_ids.sort_unstable();
    clip_ids.dedup();
    clip_ids
}

pub fn clip_library_resample_selected_model(
    world: &mut crate::ecs::world::World,
    assets: &mut AssetStorage,
    fps: f32,
) -> Option<usize> {
    use crate::animation::editable::clip_resample_at_fps;
    use crate::ecs::resource::EditHistory;

    let clip_ids = query_selected_model_clip_ids(world);
    if clip_ids.is_empty() {
        return None;
    }

    let mut new_sources: Vec<(SourceClipId, SourceClip)> = Vec::new();
    {
        let mut lib = world.resource_mut::<ClipLibrary>();
        for source_id in &clip_ids {
            let Some(mut clip) = lib.get(*source_id).cloned() else {
                continue;
            };
            let summary = clip_resample_at_fps(&mut clip, fps);
            clip.name = format!("{} ({}fps)", clip.name, fps as i64);
            let new_id = clip_library_register_and_activate(&mut lib, assets, clip);

            log!(
                "Resampled clip (src {}) to new clip {} at {} fps: {} curves, {} keyframes",
                source_id,
                new_id,
                fps,
                summary.resampled_curve_count,
                summary.total_keyframe_count,
            );
            if let Some(source) = lib.source_clips.get(&new_id).cloned() {
                new_sources.push((new_id, source));
            }
        }
    }

    if world.contains_resource::<EditHistory>() {
        let mut edit_history = world.resource_mut::<EditHistory>();
        for (new_id, source) in &new_sources {
            edit_history.push_clip_added(*new_id, source.clone(), "resample clip");
        }
    }

    Some(new_sources.len())
}

pub fn clip_library_count_references(
    world: &crate::ecs::world::World,
) -> Vec<(SourceClipId, usize)> {
    use crate::ecs::component::ClipSchedule;

    let mut counts: HashMap<SourceClipId, usize> = HashMap::new();

    let entities = world.component_entities::<ClipSchedule>();
    for entity in entities {
        if let Some(schedule) = world.get_component::<ClipSchedule>(entity) {
            for inst in &schedule.instances {
                *counts.entry(inst.source_id).or_insert(0) += 1;
            }
        }
    }

    counts.into_iter().collect()
}

fn deserialize_clip(content: &str) -> Result<EditableAnimationClip> {
    if let Ok(clip_file) = ron::from_str::<AnimationClipFile>(content) {
        return Ok(clip_file.clip);
    }

    let clip = ron::from_str::<EditableAnimationClip>(content)?;
    Ok(clip)
}
