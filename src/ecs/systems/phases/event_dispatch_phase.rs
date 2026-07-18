use cgmath::Vector3;

use crate::asset::AssetStorage;
use crate::ecs::events::UIEvent;
use crate::ecs::world::World;
use crate::ecs::UIEventQueue;

use super::super::ui_event_systems::DeferredAction;
use super::dispatch_camera::dispatch_camera_light_debug_events;
use super::dispatch_clip_browser::dispatch_clip_browser_ecs_events;
use super::dispatch_clip_instance::dispatch_clip_instance_events;
use super::dispatch_constraint::{
    dispatch_constraint_bake_events, dispatch_constraint_edit_events,
    dispatch_debug_constraint_events,
};
use super::dispatch_edit_history::dispatch_edit_history_events;
use super::dispatch_hierarchy::dispatch_hierarchy_events;
use super::dispatch_overlay::dispatch_overlay_events;
use super::dispatch_pose_library::dispatch_pose_library_events;
use super::dispatch_scene::dispatch_scene_events;
use super::dispatch_spring_bone::{
    dispatch_spring_bone_bake_ecs_events, dispatch_spring_bone_edit_events,
};
use super::dispatch_timeline::{
    dispatch_buffer_events, dispatch_keyframe_clipboard_events, dispatch_timeline_events,
};

pub fn run_event_dispatch_phase(
    world: &mut World,
    assets: &mut AssetStorage,
    model_bounds: Option<(Vector3<f32>, Vector3<f32>, Vector3<f32>)>,
) -> (Vec<UIEvent>, Vec<DeferredAction>) {
    let mut deferred: Vec<DeferredAction> = Vec::new();

    #[cfg(feature = "text-to-motion")]
    super::dispatch_ml::drain_grpc_responses(world, assets, &mut deferred);

    #[cfg(feature = "auto-rig")]
    super::dispatch_ml::poll_mesh_server_status(world);
    #[cfg(feature = "auto-rig")]
    super::dispatch_ml::poll_rigging_server_status(world);

    let events: Vec<UIEvent> = {
        if let Some(mut ui_events) = world.get_resource_mut::<UIEventQueue>() {
            ui_events.drain().collect()
        } else {
            return (Vec::new(), deferred);
        }
    };

    if events.is_empty() {
        return (Vec::new(), deferred);
    }

    let hierarchy_deferred = dispatch_hierarchy_events(&events, world, assets);
    dispatch_timeline_events(&events, world, assets);
    dispatch_keyframe_clipboard_events(&events, world);
    dispatch_buffer_events(&events, world);
    dispatch_clip_instance_events(&events, world);
    dispatch_clip_browser_ecs_events(&events, world, assets);
    dispatch_edit_history_events(&events, world);
    dispatch_scene_events(&events, world);
    dispatch_overlay_events(&events, world);
    dispatch_debug_constraint_events(&events, world, assets);
    dispatch_constraint_edit_events(&events, world);
    dispatch_constraint_bake_events(&events, world, assets);
    dispatch_pose_library_events(&events, world, assets);
    dispatch_spring_bone_bake_ecs_events(&events, world, assets);
    dispatch_spring_bone_edit_events(&events, world, assets);
    #[cfg(feature = "ml")]
    super::dispatch_ml::dispatch_curve_suggestion_events(&events, world, assets);
    #[cfg(feature = "auto-rig")]
    super::dispatch_ml::dispatch_text_to_animation_events(&events, world, assets);
    #[cfg(feature = "auto-rig")]
    super::dispatch_ml::dispatch_model_loaded_for_animation(&events, world);

    let camera_deferred = dispatch_camera_light_debug_events(&events, world, model_bounds);
    deferred.extend(camera_deferred);
    deferred.extend(hierarchy_deferred);

    #[cfg(feature = "auto-rig")]
    super::dispatch_ml::dispatch_text_to_mesh_events(&events, world, &mut deferred);
    #[cfg(feature = "auto-rig")]
    super::dispatch_ml::dispatch_auto_rig_events(&events, world, &mut deferred);

    let platform_events = filter_platform_events(&events);

    (platform_events, deferred)
}

fn filter_platform_events(events: &[UIEvent]) -> Vec<UIEvent> {
    events
        .iter()
        .filter(|e| {
            matches!(
                e,
                UIEvent::ClipBrowserLoadFromFile
                    | UIEvent::ClipBrowserSaveToFile(_)
                    | UIEvent::ClipBrowserExportFbx(_)
                    | UIEvent::ClipBrowserExportGltf(_)
                    | UIEvent::ClipBrowserExportGltfAnimationOnly(_)
                    | UIEvent::ExportModelGltf
                    | UIEvent::SpringBoneSaveBake
            )
        })
        .cloned()
        .collect()
}
