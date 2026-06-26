use std::ffi::c_void;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::{anyhow, Result};
use cgmath::{SquareMatrix, Vector4};
use vulkanalia::prelude::v1_0::*;

use crate::animation::editable::SourceClipId;
use crate::app::AppData;
use crate::asset::{AssetStorage, MeshAsset, NodeAsset, SkeletonAsset};
use crate::ecs::component::{AnimationMeta, ClipSchedule, EntityIcon};
use crate::ecs::resource::billboard::BillboardData;
use crate::ecs::resource::gizmo::{BoneGizmoData, ConstraintGizmoData};
use crate::ecs::resource::{
    AnimationType, ClipLibrary, FbxModelCache, GltfModelCache, MeshAssets, ModelState, NodeAssets,
    TimelineState,
};
use crate::ecs::world::{Animator, Transform, World};
use crate::loader::fbx::FbxModel;
use crate::loader::load_png_image;
use crate::loader::{ModelLoadResult, TextureSource};
use crate::render::MaterialUBO;
use crate::vulkanr::buffer::{RRIndexBuffer, RRVertexBuffer};
use crate::vulkanr::command::RRCommandPool;
use crate::vulkanr::data as vulkan_data;
use crate::vulkanr::data::VertexData;
use crate::vulkanr::device::RRDevice;
use crate::vulkanr::resource::graphics_resource::{
    GraphicsResources, MaterialId, MeshBuffer, NodeData,
};
use crate::vulkanr::swapchain::RRSwapchain;
use crate::vulkanr::vulkan::Instance;
use thyllore_vulkan_core::raytracing::RRAccelerationStructure;
use thyllore_vulkan_core::resource::image::{
    create_image_view, create_texture_image_pixel, create_texture_sampler,
};
use thyllore_vulkan_core::resource::raytracing_data::RayTracingData;

pub unsafe fn load_model_from_file_system(
    path: &str,
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    swapchain: &RRSwapchain,
    graphics: &mut GraphicsResources,
    raytracing: &mut RayTracingData,
    world: &mut World,
    assets: &mut AssetStorage,
    scene_will_provide_clips: bool,
) -> Result<()> {
    log!("=== Loading model from path: {} ===", path);

    let (load_result, fbx_model) = load_model_data(path)?;

    let _parent_entity = apply_model_to_resources(
        &load_result,
        path,
        instance,
        device,
        command_pool,
        swapchain,
        graphics,
        raytracing,
        world,
        assets,
        scene_will_provide_clips,
        fbx_model,
    )?;

    log!("=== Model loaded successfully ===");
    Ok(())
}

#[cfg(feature = "auto-rig")]
pub unsafe fn load_model_from_file_system_with_result(
    load_result: &ModelLoadResult,
    model_name: &str,
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    swapchain: &RRSwapchain,
    graphics: &mut GraphicsResources,
    raytracing: &mut RayTracingData,
    world: &mut World,
    assets: &mut AssetStorage,
    scene_will_provide_clips: bool,
    fbx_model: Option<FbxModel>,
) -> Result<crate::ecs::world::Entity> {
    let parent_entity = apply_model_to_resources(
        load_result,
        model_name,
        instance,
        device,
        command_pool,
        swapchain,
        graphics,
        raytracing,
        world,
        assets,
        scene_will_provide_clips,
        fbx_model,
    )?;

    log!("=== Model loaded successfully ===");
    Ok(parent_entity)
}

unsafe fn load_model_data(path: &str) -> Result<(ModelLoadResult, Option<FbxModel>)> {
    let extension = Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
        .unwrap_or_default();

    match extension.as_str() {
        "fbx" => {
            let (result, fbx_model) = crate::loader::fbx::load_fbx_to_graphics_resources(path)?;
            Ok((ModelLoadResult::from_fbx(result), Some(fbx_model)))
        }
        "gltf" | "glb" => {
            let result = crate::loader::gltf::load_gltf_file(path)?;
            Ok((ModelLoadResult::from_gltf(result), None))
        }
        "usdc" | "usda" | "usd" | "usdz" => {
            let result = crate::loader::usd::load_usd_file(path)?;
            Ok((ModelLoadResult::from_usd(result), None))
        }
        _ => Err(anyhow!(
            "Unsupported file format. Only FBX, glTF/GLB, and USD are supported."
        )),
    }
}

unsafe fn apply_model_to_resources(
    load_result: &ModelLoadResult,
    model_name: &str,
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    swapchain: &RRSwapchain,
    graphics: &mut GraphicsResources,
    raytracing: &mut RayTracingData,
    world: &mut World,
    assets: &mut AssetStorage,
    scene_will_provide_clips: bool,
    fbx_model: Option<FbxModel>,
) -> Result<crate::ecs::world::Entity> {
    cleanup_resources(device, graphics, raytracing, world, assets)?;
    insert_model_caches(world, model_name, fbx_model);
    ensure_graphics_capacity(load_result, instance, device, swapchain, graphics)?;

    setup_animation_system(world, load_result, assets);
    setup_nodes(world, load_result);

    for (i, loaded_mesh) in load_result.meshes.iter().enumerate() {
        let mesh_buffer = create_mesh_buffer(
            instance,
            device,
            command_pool,
            graphics,
            loaded_mesh,
            i,
            model_name,
        )?;
        let material_id = create_material_for_mesh(
            instance,
            device,
            graphics,
            &mesh_buffer,
            i,
            loaded_mesh.base_color_factor,
        )?;

        graphics.meshes.push(mesh_buffer);
        graphics.mesh_material_ids.push(material_id);
    }

    apply_initial_pose(
        instance,
        device,
        command_pool,
        graphics,
        world,
        assets,
        load_result,
    )?;
    rebuild_acceleration_structures(instance, device, command_pool, graphics, raytracing)?;
    update_ray_query_descriptor(device, raytracing)?;

    {
        let mut billboard = world.resource_mut::<BillboardData>();
        update_billboard_descriptor(device, swapchain, &mut *billboard)?;
    }

    let animation_type = determine_animation_type(load_result);
    let node_animation_scale = load_result.node_animation_scale;
    log_model_load_info(load_result, animation_type.clone(), node_animation_scale);

    let parent_entity = create_ecs_entities(
        model_name,
        graphics,
        world,
        assets,
        animation_type,
        node_animation_scale,
        &load_result.clips.clone(),
        scene_will_provide_clips,
    );

    let path_lower = model_name.to_lowercase();
    if path_lower.ends_with(".gltf") || path_lower.ends_with(".glb") || path_lower.ends_with(".fbx")
    {
        world.insert_component(
            parent_entity,
            crate::ecs::component::GlbSource::FilePath(model_name.to_string()),
        );
    }

    apply_loaded_constraints(load_result, world);
    apply_loaded_spring_bones(load_result, world);
    initialize_bone_gizmo_visibility(
        world,
        assets,
        graphics,
        node_animation_scale,
        load_result.has_skinned_meshes,
    );
    initialize_constraint_gizmo_visibility(world);
    setup_curves(world, &load_result.curves);
    setup_point_cloud(world, &load_result.points);

    Ok(parent_entity)
}

fn setup_curves(world: &mut World, curves: &[thyllore_importer_core::UsdCurveData]) {
    if !world.contains_resource::<crate::ecs::resource::CurveMeshData>() {
        return;
    }

    let mut curve_mesh = world.resource_mut::<crate::ecs::resource::CurveMeshData>();
    crate::ecs::systems::build_curve_line_mesh(curves, &mut curve_mesh.mesh);
    curve_mesh.visible = !curve_mesh.mesh.indices.is_empty();
    curve_mesh.dirty = true;
}

fn setup_point_cloud(world: &mut World, points: &[thyllore_importer_core::UsdPointData]) {
    if !world.contains_resource::<crate::ecs::resource::PointCloudData>() {
        return;
    }

    let mut point_cloud = world.resource_mut::<crate::ecs::resource::PointCloudData>();
    crate::ecs::systems::build_point_cloud_mesh(points, &mut point_cloud.mesh);
    point_cloud.visible = !point_cloud.mesh.indices.is_empty();
    point_cloud.dirty = true;
}

fn insert_model_caches(world: &mut World, model_name: &str, fbx_model: Option<FbxModel>) {
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

unsafe fn ensure_graphics_capacity(
    load_result: &ModelLoadResult,
    instance: &Instance,
    device: &RRDevice,
    swapchain: &RRSwapchain,
    graphics: &mut GraphicsResources,
) -> Result<()> {
    let mesh_count = load_result.meshes.len();
    let reserved_scene_objects = 4;
    let required_materials = mesh_count as u32 + reserved_scene_objects as u32;
    let required_objects = graphics.objects.get_next_slot() + mesh_count + reserved_scene_objects;

    graphics
        .materials
        .ensure_capacity(device, required_materials)?;
    graphics.objects.ensure_capacity(
        instance,
        device,
        swapchain.swapchain_images.len(),
        required_objects,
    )?;

    Ok(())
}

fn determine_animation_type(load_result: &ModelLoadResult) -> AnimationType {
    if load_result.has_skinned_meshes {
        AnimationType::Skeletal
    } else if !load_result.clips.is_empty() {
        AnimationType::Node
    } else {
        AnimationType::None
    }
}

fn log_model_load_info(
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

unsafe fn cleanup_resources(
    device: &RRDevice,
    graphics: &mut GraphicsResources,
    raytracing: &mut RayTracingData,
    world: &mut World,
    assets: &mut AssetStorage,
) -> Result<()> {
    log!("Cleaning up model resources...");

    device.device.device_wait_idle()?;

    if let Some(ref mut accel) = raytracing.acceleration_structure {
        accel.destroy(&device.device);
    }
    raytracing.acceleration_structure = None;

    graphics.clear_meshes(device);
    graphics.mesh_material_ids.clear();
    graphics.materials.clear_materials(&device.device);
    graphics.objects.reset_to_reserved();

    if world.contains_resource::<crate::ecs::resource::CurveMeshData>() {
        let mut curves = world.resource_mut::<crate::ecs::resource::CurveMeshData>();
        curves.mesh.vertices.clear();
        curves.mesh.indices.clear();
        curves.visible = false;
        curves.dirty = true;
    }

    if world.contains_resource::<crate::ecs::resource::PointCloudData>() {
        let mut points = world.resource_mut::<crate::ecs::resource::PointCloudData>();
        points.mesh.vertices.clear();
        points.mesh.indices.clear();
        points.visible = false;
        points.dirty = true;
    }

    if world.contains_resource::<ClipLibrary>() {
        let mut clip_library = world.resource_mut::<ClipLibrary>();
        clip_library.clear();
    }

    if world.contains_resource::<TimelineState>() {
        let mut timeline_state = world.resource_mut::<TimelineState>();
        timeline_state.current_clip_id = None;
        timeline_state.current_time = 0.0;
        timeline_state.selected_keyframes.clear();
        timeline_state.expanded_tracks.clear();
    }

    if world.contains_resource::<FbxModelCache>() {
        let mut cache = world.resource_mut::<FbxModelCache>();
        cache.clear();
    }

    if world.contains_resource::<GltfModelCache>() {
        let mut cache = world.resource_mut::<GltfModelCache>();
        cache.clear();
    }

    if world.contains_resource::<MeshAssets>() {
        let mut mesh_assets = world.resource_mut::<MeshAssets>();
        mesh_assets.meshes.clear();
    }

    if world.contains_resource::<NodeAssets>() {
        let mut node_assets = world.resource_mut::<NodeAssets>();
        node_assets.nodes.clear();
    }

    if world.contains_resource::<crate::ecs::resource::BonePoseOverride>() {
        let mut overrides = world.resource_mut::<crate::ecs::resource::BonePoseOverride>();
        overrides.overrides.clear();
    }

    world.clear();
    assets.clear();

    log!("Model resources cleaned up");
    Ok(())
}

fn setup_animation_system(
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

fn setup_nodes(world: &mut World, load_result: &ModelLoadResult) {
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

unsafe fn create_mesh_buffer(
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    graphics: &mut GraphicsResources,
    loaded_mesh: &crate::loader::LoadedMesh,
    mesh_index: usize,
    model_path: &str,
) -> Result<MeshBuffer> {
    let mut mesh = MeshBuffer::default();

    match &loaded_mesh.texture {
        Some(TextureSource::Embedded(tex)) => {
            (mesh.image, mesh.image_memory, mesh.mip_level) = create_texture_image_pixel(
                instance,
                device,
                command_pool,
                &tex.data,
                tex.width,
                tex.height,
            )?;
        }
        Some(TextureSource::File(texture_path)) => {
            let resolved = resolve_texture_path(texture_path, model_path);
            let load_path = resolved.to_string_lossy();
            match load_png_image(&load_path) {
                Ok((image_data, width, height)) => {
                    (mesh.image, mesh.image_memory, mesh.mip_level) = create_texture_image_pixel(
                        instance,
                        device,
                        command_pool,
                        &image_data,
                        width,
                        height,
                    )?;
                }
                Err(e) => {
                    log_warn!("Failed to load texture {}: {}", load_path, e);
                    let white_pixel = vec![255u8, 255, 255, 255];
                    (mesh.image, mesh.image_memory, mesh.mip_level) = create_texture_image_pixel(
                        instance,
                        device,
                        command_pool,
                        &white_pixel,
                        1,
                        1,
                    )?;
                }
            }
        }
        None => {
            let white_pixel = vec![255u8, 255, 255, 255];
            (mesh.image, mesh.image_memory, mesh.mip_level) =
                create_texture_image_pixel(instance, device, command_pool, &white_pixel, 1, 1)?;
        }
    }

    mesh.image_view = create_image_view(
        device,
        mesh.image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
        mesh.mip_level,
    )?;
    mesh.sampler = create_texture_sampler(device, mesh.mip_level)?;

    mesh.vertex_data = loaded_mesh.vertex_data.clone();
    mesh.skin_data = loaded_mesh.skin_data.clone();
    mesh.skeleton_id = loaded_mesh.skeleton_id;
    mesh.node_index = loaded_mesh.node_index;
    mesh.base_vertices = loaded_mesh.local_vertices.clone();
    mesh.base_colors = Some(
        mesh.vertex_data
            .vertices
            .iter()
            .map(|v| cgmath::Vector4::new(v.color.x, v.color.y, v.color.z, v.color.w))
            .collect(),
    );

    mesh.vertex_buffer = RRVertexBuffer::new(
        instance,
        device,
        command_pool,
        (size_of::<vulkan_data::Vertex>() * mesh.vertex_data.vertices.len()) as vk::DeviceSize,
        mesh.vertex_data.vertices.as_ptr() as *const c_void,
        mesh.vertex_data.vertices.len(),
    )?;

    mesh.index_buffer = RRIndexBuffer::new(
        instance,
        device,
        command_pool,
        (size_of::<u32>() * mesh.vertex_data.indices.len()) as u64,
        mesh.vertex_data.indices.as_ptr() as *const c_void,
        mesh.vertex_data.indices.len(),
    )?;

    mesh.object_index = graphics.objects.allocate_slot();
    log!(
        "Allocated object_index {} for mesh {}",
        mesh.object_index,
        mesh_index
    );

    Ok(mesh)
}

unsafe fn create_material_for_mesh(
    instance: &Instance,
    device: &RRDevice,
    graphics: &mut GraphicsResources,
    mesh: &MeshBuffer,
    mesh_index: usize,
    base_color_factor: [f32; 4],
) -> Result<MaterialId> {
    let material_name = format!("material_{}", mesh_index);
    let material_properties = MaterialUBO {
        base_color: Vector4::new(
            base_color_factor[0],
            base_color_factor[1],
            base_color_factor[2],
            base_color_factor[3],
        ),
        ..MaterialUBO::default()
    };

    let material_id = graphics.materials.create_material_with_texture(
        instance,
        device,
        &material_name,
        mesh.image_view,
        mesh.sampler,
        material_properties,
    )?;

    log!("Created material {} for mesh {}", material_id, mesh_index);
    Ok(material_id)
}

unsafe fn apply_initial_pose(
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    graphics: &mut GraphicsResources,
    world: &mut World,
    assets: &AssetStorage,
    load_result: &ModelLoadResult,
) -> Result<()> {
    use crate::ecs::{compute_pose_global_transforms, create_pose_from_rest, sample_clip_to_pose};

    if load_result.clips.is_empty() {
        return Ok(());
    }

    log!("Applying initial pose (time=0) for animation...");

    if !load_result.clips.is_empty() {
        if world.contains_resource::<TimelineState>() {
            let mut timeline = world.resource_mut::<TimelineState>();
            timeline.playing = false;
            timeline.current_time = 0.0;
        }
    }

    let skeleton_id = graphics.meshes.first().and_then(|m| m.skeleton_id);

    if let Some(skel_id) = skeleton_id {
        let (current_time, looping) = if world.contains_resource::<TimelineState>() {
            let timeline = world.resource::<TimelineState>();
            (timeline.current_time, timeline.looping)
        } else {
            (0.0, true)
        };
        let skeleton = assets.get_skeleton_by_skeleton_id(skel_id);
        let first_clip = assets.animation_clips.values().next().map(|a| &a.clip);

        if let (Some(skeleton), Some(clip)) = (skeleton, first_clip) {
            let mut pose = create_pose_from_rest(skeleton);
            sample_clip_to_pose(clip, current_time, skeleton, &mut pose, looping);
            let globals = compute_pose_global_transforms(skeleton, &pose);
            let skeleton_clone = skeleton.clone();

            for mesh_idx in 0..graphics.meshes.len() {
                apply_skinning_to_mesh(
                    instance,
                    device,
                    command_pool,
                    graphics,
                    &globals,
                    &skeleton_clone,
                    mesh_idx,
                )?;
            }
        }
    }

    let has_node_animation = !load_result.has_skinned_meshes && !graphics.meshes.is_empty();
    if has_node_animation {
        let mut node_assets = world.resource_mut::<NodeAssets>();
        let node_animation_scale = load_result.node_animation_scale;

        let skel_id = graphics.meshes.first().and_then(|m| m.skeleton_id);
        let skeleton_clone = skel_id.and_then(|id| assets.get_skeleton_by_skeleton_id(id).cloned());
        let clip_clone = assets
            .animation_clips
            .values()
            .next()
            .map(|a| a.clip.clone());

        let updated_meshes = if let (Some(skeleton), Some(clip)) = (&skeleton_clone, &clip_clone) {
            let mut pose = create_pose_from_rest(skeleton);
            sample_clip_to_pose(clip, 0.0, skeleton, &mut pose, false);

            crate::ecs::systems::animation::apply::prepare_node_animation(
                graphics,
                &mut node_assets.nodes,
                skeleton,
                &pose,
                node_animation_scale,
            )
        } else {
            Vec::new()
        };

        for mesh_idx in updated_meshes {
            if let Err(e) = upload_mesh_vertices(instance, device, command_pool, graphics, mesh_idx)
            {
                log!(
                    "Failed to upload initial node animation mesh {}: {}",
                    mesh_idx,
                    e
                );
            }
        }
    }

    log!("Initial pose applied successfully");
    Ok(())
}

unsafe fn apply_skinning_to_mesh(
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    graphics: &mut GraphicsResources,
    global_transforms: &[cgmath::Matrix4<f32>],
    skeleton: &crate::animation::Skeleton,
    mesh_idx: usize,
) -> Result<()> {
    use crate::ecs::systems::animation::apply::apply_skinning_to_single_mesh;

    if !apply_skinning_to_single_mesh(graphics, mesh_idx, global_transforms, skeleton) {
        return Ok(());
    }

    let mesh = &mut graphics.meshes[mesh_idx];
    if let Err(e) = mesh.vertex_buffer.update(
        instance,
        device,
        command_pool,
        (size_of::<vulkan_data::Vertex>() * mesh.vertex_data.vertices.len()) as vk::DeviceSize,
        mesh.vertex_data.vertices.as_ptr() as *const c_void,
        mesh.vertex_data.vertices.len(),
    ) {
        log!(
            "Failed to update vertex buffer for mesh {}: {}",
            mesh_idx,
            e
        );
    }

    Ok(())
}

unsafe fn upload_mesh_vertices(
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    graphics: &mut GraphicsResources,
    mesh_idx: usize,
) -> Result<()> {
    if mesh_idx >= graphics.meshes.len() {
        return Ok(());
    }

    let mesh = &mut graphics.meshes[mesh_idx];
    let vertices = &mesh.vertex_data.vertices;
    let vertex_count = vertices.len();
    let vertex_stride = size_of::<vulkan_data::Vertex>();

    mesh.vertex_buffer.update(
        instance,
        device,
        command_pool,
        (vertex_stride * vertex_count) as vk::DeviceSize,
        vertices.as_ptr() as *const c_void,
        vertex_count,
    )?;

    Ok(())
}

pub unsafe fn rebuild_acceleration_structures(
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    graphics: &GraphicsResources,
    raytracing: &mut RayTracingData,
) -> Result<()> {
    log!("Rebuilding acceleration structures...");

    let mut acceleration_structure = RRAccelerationStructure::new();

    for mesh in &graphics.meshes {
        if !mesh.render_to_gbuffer {
            continue;
        }

        let blas = RRAccelerationStructure::create_blas(
            instance,
            device,
            command_pool.as_ref(),
            &mesh.vertex_buffer.buffer,
            mesh.vertex_data.vertices.len() as u32,
            std::mem::size_of::<vulkan_data::Vertex>() as u32,
            &mesh.index_buffer.buffer,
            mesh.vertex_data.indices.len() as u32,
        )?;

        acceleration_structure.blas_list.push(blas);
        log!("Created BLAS for mesh");
    }

    if !acceleration_structure.blas_list.is_empty() {
        let tlas = RRAccelerationStructure::create_tlas(
            instance,
            device,
            command_pool.as_ref(),
            &acceleration_structure.blas_list,
        )?;
        acceleration_structure.tlas = tlas;
        log!(
            "Created TLAS with {} instances",
            acceleration_structure.blas_list.len()
        );
    }

    if acceleration_structure.blas_list.is_empty() {
        raytracing.acceleration_structure = None;
    } else {
        raytracing.acceleration_structure = Some(acceleration_structure);
    }
    log!("Acceleration structures rebuilt successfully");
    Ok(())
}

pub unsafe fn rebuild_acceleration_structures_from_data(
    instance: &Instance,
    rrdevice: &RRDevice,
    data: &mut AppData,
    rrcommand_pool: &Rc<RRCommandPool>,
) -> Result<()> {
    rebuild_acceleration_structures(
        instance,
        rrdevice,
        rrcommand_pool,
        &data.graphics_resources,
        &mut data.raytracing,
    )
}

unsafe fn update_ray_query_descriptor(
    device: &RRDevice,
    raytracing: &mut RayTracingData,
) -> Result<()> {
    if let Some(ref accel_struct) = raytracing.acceleration_structure {
        if let Some(tlas) = accel_struct.tlas.acceleration_structure {
            if let Some(ref mut ray_query_desc) = raytracing.ray_query_descriptor {
                ray_query_desc.update_tlas(device, tlas)?;
                log!("Updated ray_query_descriptor with new TLAS");
            }
        }
    }
    Ok(())
}

unsafe fn update_billboard_descriptor(
    device: &RRDevice,
    swapchain: &RRSwapchain,
    billboard: &mut BillboardData,
) -> Result<()> {
    let texture_clone = billboard.render_state.texture.clone();
    if let Some(ref billboard_texture) = texture_clone {
        billboard
            .render_state
            .descriptor_set
            .update_descriptor_sets(device, swapchain, billboard_texture)?;
        log!("Re-updated billboard.render_state.descriptor_set after model reload");
    }
    Ok(())
}

fn create_ecs_entities(
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

fn ensure_ecs_resources(world: &mut World) {
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

fn initialize_bone_gizmo_visibility(
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

fn apply_loaded_constraints(load_result: &ModelLoadResult, world: &mut World) {
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

fn apply_loaded_spring_bones(load_result: &ModelLoadResult, world: &mut World) {
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

fn initialize_constraint_gizmo_visibility(world: &mut World) {
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

fn resolve_texture_path(texture_path: &str, model_path: &str) -> PathBuf {
    let original = Path::new(texture_path);
    if original.exists() {
        return original.to_path_buf();
    }

    let file_stem = original.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let file_name = original.file_name().and_then(|s| s.to_str()).unwrap_or("");

    let model_dir = Path::new(model_path)
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let model_root = model_dir.parent().unwrap_or(model_dir);

    let texture_dir = original.parent().unwrap_or_else(|| Path::new("."));
    let texture_root = texture_dir.parent().unwrap_or(texture_dir);

    let mut search_dirs = vec![
        model_dir.to_path_buf(),
        model_dir.join("textures"),
        model_root.join("textures"),
    ];

    if texture_dir != model_dir {
        search_dirs.push(texture_dir.to_path_buf());
        search_dirs.push(texture_dir.join("textures"));
        search_dirs.push(texture_root.join("textures"));
    }

    let candidate_names: Vec<String> = vec![
        file_name.to_string(),
        format!("{}.png", file_name),
        format!("{}.png", file_stem),
        format!("{}.jpg", file_stem),
    ];

    for dir in &search_dirs {
        for name in &candidate_names {
            let candidate = dir.join(name);
            if candidate.exists() {
                log!(
                    "Resolved texture: {} -> {}",
                    texture_path,
                    candidate.display()
                );
                return candidate;
            }
        }
    }

    log!("Texture not found, using original path: {}", texture_path);
    original.to_path_buf()
}

pub unsafe fn load_model_additive(
    path: &str,
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    swapchain: &RRSwapchain,
    graphics: &mut GraphicsResources,
    raytracing: &mut RayTracingData,
    world: &mut World,
    assets: &mut AssetStorage,
) -> Result<()> {
    let (load_result, _fbx_model) = load_model_data(path)?;

    let part_name = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("part")
        .to_string();

    let parent_entity = append_model_to_scene(
        &load_result,
        &part_name,
        instance,
        device,
        command_pool,
        swapchain,
        graphics,
        raytracing,
        world,
        assets,
    )?;

    world.insert_component(
        parent_entity,
        crate::ecs::component::GlbSource::FilePath(path.to_string()),
    );

    Ok(())
}

unsafe fn append_model_to_scene(
    load_result: &ModelLoadResult,
    part_name: &str,
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    swapchain: &RRSwapchain,
    graphics: &mut GraphicsResources,
    raytracing: &mut RayTracingData,
    world: &mut World,
    assets: &mut AssetStorage,
) -> Result<crate::ecs::world::Entity> {
    ensure_graphics_capacity(load_result, instance, device, swapchain, graphics)?;

    let mesh_index_offset = graphics.meshes.len();

    for (i, loaded_mesh) in load_result.meshes.iter().enumerate() {
        let global_index = mesh_index_offset + i;
        let mesh_buffer = create_mesh_buffer(
            instance,
            device,
            command_pool,
            graphics,
            loaded_mesh,
            global_index,
            part_name,
        )?;
        let material_id = create_material_for_mesh(
            instance,
            device,
            graphics,
            &mesh_buffer,
            global_index,
            loaded_mesh.base_color_factor,
        )?;

        graphics.meshes.push(mesh_buffer);
        graphics.mesh_material_ids.push(material_id);
    }

    rebuild_acceleration_structures(instance, device, command_pool, graphics, raytracing)?;
    update_ray_query_descriptor(device, raytracing)?;

    {
        let mut billboard = world.resource_mut::<BillboardData>();
        update_billboard_descriptor(device, swapchain, &mut *billboard)?;
    }

    ensure_ecs_resources(world);

    let parent_entity = world
        .entity()
        .with_name(part_name)
        .with_transform(Transform::default())
        .with_visible(true)
        .with_editor_display(EntityIcon::Model, true)
        .build();

    log!(
        "Created additive parent entity '{}': entity_id={}",
        part_name,
        parent_entity
    );

    build_mesh_entities_range(
        part_name,
        graphics,
        world,
        assets,
        parent_entity,
        mesh_index_offset..graphics.meshes.len(),
    );

    log!(
        "Additively loaded '{}': {} meshes, total entities={}",
        part_name,
        load_result.meshes.len(),
        world.entity_count()
    );

    Ok(parent_entity)
}

fn build_mesh_entities_range(
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
