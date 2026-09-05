mod apply;
mod gpu_upload;
mod scene_model;

pub use scene_model::{build_initial_clip_schedule, find_best_clip};

use std::rc::Rc;

use anyhow::{anyhow, Result};

use crate::asset::AssetStorage;
use crate::ecs::component::EntityIcon;
use crate::ecs::resource::billboard::BillboardData;
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

    let _parent_entity = apply::apply_model_to_resources(
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
    let parent_entity = apply::apply_model_to_resources(
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

    let waters = crate::app::raytracing::scene_build::collect_water_instances(world);
    let mesh_transforms =
        crate::app::raytracing::scene_build::collect_mesh_transforms(world, assets);
    crate::app::raytracing::scene_build::rebuild_acceleration_structures(
        instance,
        device,
        command_pool,
        graphics,
        raytracing,
        &waters,
        &mesh_transforms,
    )?;
    crate::app::raytracing::scene_build::update_ray_query_descriptor(device, raytracing)?;

    {
        let mut billboard = world.resource_mut::<BillboardData>();
        crate::app::raytracing::scene_build::update_billboard_descriptor(
            device,
            swapchain,
            &mut *billboard,
        )?;
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
