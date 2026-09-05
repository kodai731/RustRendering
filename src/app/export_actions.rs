use crate::app::App;
use crate::ecs::resource::ClipLibrary;

pub(crate) fn handle_clip_export_fbx(app: &mut App, source_id: u64) {
    let clip = {
        let lib = app.data.ecs_world.resource::<ClipLibrary>();
        lib.get(source_id).cloned()
    };
    let skeleton = app
        .data
        .ecs_assets
        .skeletons
        .values()
        .next()
        .map(|sa| sa.skeleton.clone());

    let (Some(clip), Some(skeleton)) = (clip, skeleton) else {
        return;
    };

    let default_filename = format!("{}.fbx", clip.name);
    let path = rfd::FileDialog::new()
        .add_filter("FBX Binary", &["fbx"])
        .set_file_name(&default_filename)
        .save_file();

    let Some(path) = path else {
        return;
    };

    let has_fbx_cache = app
        .data
        .ecs_world
        .contains_resource::<crate::ecs::resource::FbxModelCache>();
    let (fbx_model_ref, needs_coord_conversion) = if has_fbx_cache {
        let cache = app
            .data
            .ecs_world
            .resource::<crate::ecs::resource::FbxModelCache>();
        (cache.fbx_model().cloned(), cache.needs_coord_conversion())
    } else {
        (None, false)
    };

    let (axes, fps) = if let Some(ref fbx_model) = fbx_model_ref {
        (fbx_model.axes.clone(), fbx_model.fps)
    } else {
        (crate::loader::fbx::fbx::FbxAxesInfo::default(), 24.0)
    };

    let result = if let Some(ref fbx_model) = fbx_model_ref {
        crate::exporter::fbx_exporter::export_full_fbx(fbx_model, Some(&clip), &skeleton, &path)
    } else {
        crate::exporter::fbx_animation::export_animation_fbx(
            &clip,
            &skeleton,
            &path,
            needs_coord_conversion,
            axes,
            fps,
        )
    };

    match result {
        Ok(()) => msg_info!("FBX exported: {:?}", path),
        Err(e) => msg_error!("FBX export failed: {:?}", e),
    }
}

pub(crate) fn handle_clip_export_gltf(app: &mut App, source_id: u64) {
    let clip = {
        let lib = app.data.ecs_world.resource::<ClipLibrary>();
        lib.get(source_id).cloned()
    };
    let skeleton = app
        .data
        .ecs_assets
        .skeletons
        .values()
        .next()
        .map(|sa| sa.skeleton.clone());

    let (Some(clip), Some(skeleton)) = (clip, skeleton) else {
        return;
    };

    let source_bytes = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::GltfModelCache>()
        .and_then(|cache| resolve_glb_bytes(&*cache));

    let Some(source_bytes) = source_bytes else {
        msg_error!("glTF export failed: no source glTF/GLB model loaded");
        return;
    };

    let default_filename = format!("{}.glb", clip.name);
    let path = rfd::FileDialog::new()
        .add_filter("glTF Binary", &["glb"])
        .set_file_name(&default_filename)
        .save_file();

    let Some(path) = path else {
        return;
    };

    match crate::exporter::gltf::export_gltf_animation_from_bytes(
        &source_bytes,
        &clip,
        &skeleton,
        &path,
    ) {
        Ok(()) => msg_info!("glTF exported: {:?}", path),
        Err(e) => msg_error!("glTF export failed: {:?}", e),
    }
}

pub(crate) fn handle_clip_export_gltf_animation_only(app: &mut App, source_id: u64) {
    let clip = {
        let lib = app.data.ecs_world.resource::<ClipLibrary>();
        lib.get(source_id).cloned()
    };
    let skeleton = app.data.ecs_assets.skeletons.values().next();

    let (Some(clip), Some(skeleton)) = (clip, skeleton) else {
        return;
    };
    let default_filename = format!("{}_anim_only.glb", clip.name);
    let path = rfd::FileDialog::new()
        .add_filter("glTF Binary", &["glb"])
        .set_file_name(&default_filename)
        .save_file();

    let Some(path) = path else {
        return;
    };

    match crate::exporter::gltf::export_gltf_animation_only(&clip, &skeleton.skeleton, &path) {
        Ok(()) => msg_info!("Animation-only glTF exported: {:?}", path),
        Err(e) => msg_error!("Animation-only glTF export failed: {:?}", e),
    }
}

pub(crate) fn handle_export_model_gltf(app: &mut App) {
    let cache = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::GltfModelCache>();

    let glb_bytes = match cache {
        Some(c) => resolve_glb_bytes(&*c),
        None => None,
    };

    let Some(glb_bytes) = glb_bytes else {
        msg_error!("glTF export failed: no model data available");
        return;
    };

    let model_name = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::ModelState>()
        .map(|s| s.model_path.clone())
        .unwrap_or_else(|| "model".to_string());

    let default_filename = format!(
        "{}.glb",
        std::path::Path::new(&model_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
    );

    let path = rfd::FileDialog::new()
        .add_filter("glTF Binary", &["glb"])
        .set_file_name(&default_filename)
        .save_file();

    let Some(path) = path else {
        return;
    };

    match std::fs::write(&path, &glb_bytes) {
        Ok(()) => msg_info!("Model exported: {:?}", path),
        Err(e) => msg_error!("Model export failed: {:?}", e),
    }
}

pub(crate) fn resolve_glb_bytes(cache: &crate::ecs::resource::GltfModelCache) -> Option<Vec<u8>> {
    if let Some(ref data) = cache.glb_data {
        return Some(data.clone());
    }

    if let Some(ref path) = cache.source_path {
        return std::fs::read(path).ok();
    }

    None
}

pub(crate) fn extract_clip_name_from_path(path: &std::path::Path) -> String {
    let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("clip");

    filename
        .strip_suffix(".anim.ron")
        .or_else(|| filename.strip_suffix(".ron"))
        .unwrap_or(filename)
        .to_string()
}
