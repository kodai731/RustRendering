use crate::app::{debug_dump_actions, export_actions, App};
use crate::ecs::events::UIEvent;
use crate::ecs::resource::ClipLibrary;
use crate::ecs::DeferredAction;

pub(super) fn process_platform_file_events(
    events: &[UIEvent],
    app: &mut App,
) -> Vec<DeferredAction> {
    let mut deferred = Vec::new();

    for event in events {
        match event {
            UIEvent::ClipBrowserLoadFromFile => {
                if let Some(action) = open_clip_load_dialog() {
                    deferred.push(action);
                }
            }
            UIEvent::ClipBrowserSaveToFile(source_id) => {
                if let Some(action) = open_clip_save_dialog(app, *source_id) {
                    deferred.push(action);
                }
            }
            UIEvent::ClipBrowserExportFbx(source_id) => {
                export_actions::handle_clip_export_fbx(app, *source_id)
            }
            UIEvent::ClipBrowserExportGltf(source_id) => {
                export_actions::handle_clip_export_gltf(app, *source_id)
            }
            UIEvent::ClipBrowserExportGltfAnimationOnly(source_id) => {
                export_actions::handle_clip_export_gltf_animation_only(app, *source_id)
            }
            UIEvent::ExportModelGltf => export_actions::handle_export_model_gltf(app),
            UIEvent::SpringBoneSaveBake => {
                if let Some(action) = open_spring_bone_save_dialog(app) {
                    deferred.push(action);
                }
            }
            _ => {}
        }
    }

    deferred
}

fn open_clip_load_dialog() -> Option<DeferredAction> {
    let path = rfd::FileDialog::new()
        .add_filter("Animation RON", &["anim.ron", "ron"])
        .pick_file()?;

    Some(DeferredAction::LoadClipFromFile { path })
}

fn open_clip_save_dialog(app: &App, source_id: u64) -> Option<DeferredAction> {
    let current_name = {
        let lib = app.data.ecs_world.resource::<ClipLibrary>();
        lib.get(source_id)
            .map(|c| c.name.clone())
            .unwrap_or_else(|| "clip".to_string())
    };

    let path = rfd::FileDialog::new()
        .add_filter("Animation RON", &["anim.ron", "ron"])
        .set_file_name(format!("{}.anim.ron", current_name))
        .save_file()?;

    Some(DeferredAction::SaveClipToFile { source_id, path })
}

fn open_spring_bone_save_dialog(app: &App) -> Option<DeferredAction> {
    use crate::ecs::resource::SpringBoneState;

    let spring_state = app.data.ecs_world.resource::<SpringBoneState>();
    let baked_id = spring_state.baked_clip_source_id?;
    drop(spring_state);

    let path = rfd::FileDialog::new()
        .add_filter("Animation RON", &["anim.ron", "ron"])
        .set_file_name("spring_baked.anim.ron")
        .save_file()?;

    Some(DeferredAction::SaveSpringBoneBake { baked_id, path })
}
