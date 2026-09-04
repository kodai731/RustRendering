use super::{debug_dump, export};
use crate::app::App;
use crate::ecs::events::UIEvent;
use crate::ecs::resource::ClipLibrary;
use crate::ecs::DeferredAction;
#[cfg(feature = "auto-rig")]
use crate::ecs::UIEventQueue;

pub(super) unsafe fn execute_deferred_action(app: &mut App, action: DeferredAction) {
    match action {
        DeferredAction::LoadModel { path } => {
            if let Err(e) = app.load_model(&path) {
                log_error!("Failed to load model: {:?}", e);
            }
        }

        DeferredAction::LoadModelAdditive { path } => {
            if let Err(e) = app.load_model_additive(&path) {
                log_error!("Failed to add model: {:?}", e);
            }
        }

        DeferredAction::SpawnDebugPrimitive { kind } => {
            if let Err(e) = app.spawn_debug_primitive(kind) {
                log_error!("Failed to spawn debug primitive: {:?}", e);
            }
        }

        DeferredAction::DeleteEntities { entities } => {
            if let Err(e) = app.delete_entities(&entities) {
                log_error!("Failed to delete entities: {:?}", e);
            }
        }

        DeferredAction::TakeScreenshot => {
            log!("Taking screenshot...");
            let image_index = app.frame % crate::app::init::MAX_FRAMES_IN_FLIGHT;
            let save_result = app.save_screenshot(image_index);
            match &save_result {
                Ok(path) => msg_info!("Screenshot saved: {}", path),
                Err(e) => log_error!("Screenshot failed: {:?}", e),
            }
            crate::ecs::systems::batch_run_record_screenshot(
                &app.data.ecs_world,
                save_result.map_err(|e| format!("{e:?}")),
            );
            debug_dump::save_flame_history_npy_if_requested(app);
            debug_dump::save_water_probe_if_requested(app);
        }

        #[cfg(debug_assertions)]
        DeferredAction::DebugShadowInfo => {
            crate::debugview::log_shadow_debug_info(
                &app.data.ecs_world,
                &app.data.raytracing,
                &app.data.graphics_resources,
            );
        }

        #[cfg(debug_assertions)]
        DeferredAction::DebugBillboardDepth => {
            crate::debugview::collect_and_log_billboard_debug(
                &app.data.ecs_world,
                &app.data.raytracing,
            );
        }

        DeferredAction::DumpDebugInfo => {
            app.dump_debug_info();
        }

        DeferredAction::DumpWaterDebug => {
            app.dump_water_debug();
        }

        DeferredAction::DumpAnimationDebug => {
            let clip_library = app.data.ecs_world.resource::<ClipLibrary>();
            if let Err(e) = crate::ecs::systems::animation_debug_dump::dump_animation_debug(
                &app.data.ecs_world,
                &app.data.ecs_assets,
                &*clip_library,
            ) {
                log_warn!("Animation debug dump failed: {:?}", e);
            }
        }

        DeferredAction::LoadClipFromFile { path } => {
            let bone_name_to_id = app
                .data
                .ecs_assets
                .skeletons
                .values()
                .next()
                .map(|sa| sa.skeleton.bone_name_to_id.clone());

            let mut clip_library = app.data.ecs_world.resource_mut::<ClipLibrary>();
            match crate::ecs::systems::clip_library_systems::clip_library_load_from_file(
                &mut clip_library,
                &mut app.data.ecs_assets,
                &path,
                bone_name_to_id.as_ref(),
            ) {
                Ok(_) => {}
                Err(e) => msg_error!("Failed to load clip: {:?}", e),
            }
        }

        DeferredAction::SaveClipToFile { source_id, path } => {
            use crate::ecs::systems::clip_library_systems::{
                clip_library_save_to_file, clip_library_update_save_metadata,
            };

            let new_name = export::extract_clip_name_from_path(&path);
            let mut clip_library = app.data.ecs_world.resource_mut::<ClipLibrary>();
            clip_library_update_save_metadata(
                &mut clip_library,
                source_id,
                new_name.clone(),
                &path,
            );

            match clip_library_save_to_file(&clip_library, source_id, &path) {
                Ok(()) => msg_info!("Saved clip '{}' to {:?}", new_name, path),
                Err(e) => msg_error!("Failed to save clip: {:?}", e),
            }
        }

        DeferredAction::SaveSpringBoneBake { baked_id, path } => {
            use crate::ecs::systems::clip_library_systems::clip_library_save_to_file;

            let clip_library = app.data.ecs_world.resource::<ClipLibrary>();
            match clip_library_save_to_file(&clip_library, baked_id, &path) {
                Ok(()) => msg_info!("Saved spring bone bake to {:?}", path),
                Err(e) => msg_error!("Failed to save spring bone bake: {:?}", e),
            }
        }

        #[cfg(feature = "auto-rig")]
        DeferredAction::LoadModelFromMemory { glb_data, source } => {
            match app.load_model_from_glb(&glb_data) {
                Ok(()) => {
                    log!(
                        "DeferredAction::LoadModelFromMemory: load OK, sending ModelLoadedFromMemory({:?})",
                        source
                    );
                    let mut ui_events = app.data.ecs_world.resource_mut::<UIEventQueue>();
                    ui_events.send(UIEvent::ModelLoadedFromMemory { source });
                }
                Err(e) => {
                    log_error!("Failed to load generated mesh: {}", e);
                    let mut state = app
                        .data
                        .ecs_world
                        .resource_mut::<crate::ecs::resource::TextToMeshState>();
                    state.status = crate::ecs::resource::TextToMeshStatus::Error;
                    state.error_message = Some(format!("Failed to load GLB: {}", e));
                }
            }
        }
    }
}

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
                export::handle_clip_export_fbx(app, *source_id)
            }
            UIEvent::ClipBrowserExportGltf(source_id) => {
                export::handle_clip_export_gltf(app, *source_id)
            }
            UIEvent::ClipBrowserExportGltfAnimationOnly(source_id) => {
                export::handle_clip_export_gltf_animation_only(app, *source_id)
            }
            UIEvent::ExportModelGltf => export::handle_export_model_gltf(app),
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
