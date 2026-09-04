mod gpu_upload;
mod rt_build;
mod scene_model;

pub use rt_build::{
    collect_mesh_transforms, collect_water_instances, rebuild_acceleration_structures,
    rebuild_acceleration_structures_from_data,
};
pub use scene_model::{build_initial_clip_schedule, find_best_clip};

use std::rc::Rc;

use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_0::*;

use crate::asset::AssetStorage;
use crate::ecs::component::EntityIcon;
use crate::ecs::resource::billboard::BillboardData;
use crate::ecs::resource::{
    ClipLibrary, FbxModelCache, GltfModelCache, MeshAssets, NodeAssets, TimelineState,
};
use crate::ecs::world::{Transform, World};
use crate::loader::fbx::FbxModel;
use crate::loader::ModelLoadResult;
use crate::vulkanr::command::RRCommandPool;
use crate::vulkanr::device::RRDevice;
use crate::vulkanr::resource::graphics_resource::GraphicsResources;
use crate::vulkanr::swapchain::RRSwapchain;
use crate::vulkanr::vulkan::Instance;
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
    let path_lower = path.to_lowercase();

    if path_lower.ends_with(".fbx") {
        let (result, fbx_model) = crate::loader::fbx::load_fbx_to_graphics_resources(path)?;
        Ok((ModelLoadResult::from_fbx(result), Some(fbx_model)))
    } else if path_lower.ends_with(".gltf") || path_lower.ends_with(".glb") {
        let result = crate::loader::gltf::load_gltf_file(path)?;
        Ok((ModelLoadResult::from_gltf(result), None))
    } else {
        Err(anyhow!(
            "Unsupported file format. Only FBX and glTF/GLB are supported."
        ))
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
    scene_model::insert_model_caches(world, model_name, fbx_model);
    gpu_upload::ensure_graphics_capacity(load_result, instance, device, swapchain, graphics)?;

    scene_model::setup_animation_system(world, load_result, assets);
    scene_model::setup_nodes(world, load_result);

    for (i, loaded_mesh) in load_result.meshes.iter().enumerate() {
        let mesh_buffer = gpu_upload::create_mesh_buffer(
            instance,
            device,
            command_pool,
            graphics,
            loaded_mesh,
            i,
            model_name,
        )?;
        let material_id = gpu_upload::create_material_for_mesh(
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

    gpu_upload::apply_initial_pose(
        instance,
        device,
        command_pool,
        graphics,
        world,
        assets,
        load_result,
    )?;
    let waters = collect_water_instances(world);
    let mesh_transforms = collect_mesh_transforms(world, assets);
    rebuild_acceleration_structures(
        instance,
        device,
        command_pool,
        graphics,
        raytracing,
        &waters,
        &mesh_transforms,
    )?;
    rt_build::update_ray_query_descriptor(device, raytracing)?;

    {
        let mut billboard = world.resource_mut::<BillboardData>();
        rt_build::update_billboard_descriptor(device, swapchain, &mut *billboard)?;
    }

    let animation_type = scene_model::determine_animation_type(load_result);
    let node_animation_scale = load_result.node_animation_scale;
    scene_model::log_model_load_info(load_result, animation_type.clone(), node_animation_scale);

    let parent_entity = scene_model::create_ecs_entities(
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

    scene_model::apply_loaded_constraints(load_result, world);
    scene_model::apply_loaded_spring_bones(load_result, world);
    scene_model::initialize_bone_gizmo_visibility(
        world,
        assets,
        graphics,
        node_animation_scale,
        load_result.has_skinned_meshes,
    );
    scene_model::initialize_constraint_gizmo_visibility(world);

    Ok(parent_entity)
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
    scene_model::restore_batch_playback(world);

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

pub(crate) unsafe fn append_model_to_scene(
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
    gpu_upload::ensure_graphics_capacity(load_result, instance, device, swapchain, graphics)?;

    let mesh_index_offset = graphics.meshes.len();

    for (i, loaded_mesh) in load_result.meshes.iter().enumerate() {
        let global_index = mesh_index_offset + i;
        let mesh_buffer = gpu_upload::create_mesh_buffer(
            instance,
            device,
            command_pool,
            graphics,
            loaded_mesh,
            global_index,
            part_name,
        )?;
        let material_id = gpu_upload::create_material_for_mesh(
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

    let waters = collect_water_instances(world);
    let mesh_transforms = collect_mesh_transforms(world, assets);
    rebuild_acceleration_structures(
        instance,
        device,
        command_pool,
        graphics,
        raytracing,
        &waters,
        &mesh_transforms,
    )?;
    rt_build::update_ray_query_descriptor(device, raytracing)?;

    {
        let mut billboard = world.resource_mut::<BillboardData>();
        rt_build::update_billboard_descriptor(device, swapchain, &mut *billboard)?;
    }

    scene_model::ensure_ecs_resources(world);

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

    scene_model::build_mesh_entities_range(
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
