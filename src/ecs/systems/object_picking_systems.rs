use crate::asset::AssetStorage;
use crate::ecs::resource::CurveEditorState;
use crate::ecs::resource::{ClipLibrary, HierarchyDisplayMode, ObjectIdReadback, TimelineState};
use crate::ecs::systems::clip_track_systems::resolve_mesh_bone_id;
use crate::ecs::systems::hierarchy_systems::{
    hierarchy_deselect_all, hierarchy_select, hierarchy_toggle_selection,
};
use crate::ecs::world::{Entity, MeshRef, World};

pub fn find_entity_by_object_id(
    world: &World,
    assets: &AssetStorage,
    object_id: u32,
) -> Option<Entity> {
    if object_id == 0 {
        return None;
    }

    let mesh_index = (object_id - 1) as usize;
    let mesh_asset = assets.find_mesh_by_graphics_index(mesh_index)?;
    let target_asset_id = mesh_asset.id;

    world
        .iter_components::<MeshRef>()
        .find(|(_, mesh_ref)| mesh_ref.mesh_asset_id == target_asset_id)
        .map(|(entity, _)| entity)
}

pub fn apply_mesh_selection(
    world: &mut World,
    assets: &AssetStorage,
    readback: &mut ObjectIdReadback,
) {
    let Some(object_id) = readback.last_read_object_id.take() else {
        return;
    };

    let is_shift = readback.is_shift;
    let is_ctrl = readback.is_ctrl;

    if object_id == 0 {
        if !is_shift && !is_ctrl {
            let mut state = world.resource_mut::<crate::ecs::resource::HierarchyState>();
            hierarchy_deselect_all(&mut state);
            state.display_mode = HierarchyDisplayMode::Entities;
        }
        return;
    }

    let Some(entity) = find_entity_by_object_id(world, assets, object_id) else {
        return;
    };

    let mut state = world.resource_mut::<crate::ecs::resource::HierarchyState>();
    if is_shift || is_ctrl {
        hierarchy_toggle_selection(&mut state, entity);
    } else {
        hierarchy_select(&mut state, entity);
    }
    state.display_mode = HierarchyDisplayMode::Entities;

    sync_curve_editor_on_mesh_select(world, assets, entity);
}

fn sync_curve_editor_on_mesh_select(world: &World, assets: &AssetStorage, entity: Entity) {
    let is_open = world
        .get_resource::<CurveEditorState>()
        .map(|s| s.is_open)
        .unwrap_or(false);
    if !is_open {
        return;
    }

    let clip_library = world.resource::<ClipLibrary>();
    let source_id = world.resource::<TimelineState>().current_clip_id;
    let bone_id = resolve_mesh_bone_id(world, entity, assets, &clip_library, source_id);
    drop(clip_library);

    if let Some(bone_id) = bone_id {
        let mut editor = world.resource_mut::<CurveEditorState>();
        editor.select_bone(bone_id);
    }
}
