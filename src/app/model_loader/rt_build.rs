use std::rc::Rc;

use anyhow::Result;
use cgmath::SquareMatrix;
use vulkanalia::prelude::v1_0::*;

use crate::app::AppData;
use crate::asset::AssetStorage;
use crate::ecs::resource::billboard::BillboardData;
use crate::ecs::world::{GlobalTransform, MeshRef, World};
use crate::vulkanr::command::RRCommandPool;
use crate::vulkanr::data as vulkan_data;
use crate::vulkanr::device::RRDevice;
use crate::vulkanr::resource::graphics_resource::GraphicsResources;
use crate::vulkanr::swapchain::RRSwapchain;
use crate::vulkanr::vulkan::Instance;
use thyllore_math_core::AffineRows3x4;
use thyllore_vulkan_core::raytracing::RRAccelerationStructure;
use thyllore_vulkan_core::resource::raytracing_data::RayTracingData;

pub fn collect_water_instances(world: &World) -> Vec<(cgmath::Matrix4<f32>, f32, f32)> {
    world
        .query_waters()
        .iter()
        .filter_map(|&entity| {
            let effect = world.get_component::<crate::ecs::component::WaterTorusEffect>(entity)?;
            let ubo = thyllore_effect_core::build_water_ubo(effect, 0);
            Some((ubo.model, effect.major_radius, effect.minor_radius))
        })
        .collect()
}

pub fn collect_mesh_transforms(world: &World, assets: &AssetStorage) -> Vec<cgmath::Matrix4<f32>> {
    let indexed_transforms: Vec<(usize, cgmath::Matrix4<f32>)> = world
        .iter_components::<MeshRef>()
        .filter_map(|(entity, mesh_ref)| {
            let mesh_asset = assets.get_mesh(mesh_ref.mesh_asset_id)?;
            let model_matrix = world
                .get_component::<GlobalTransform>(entity)
                .map(|global_transform| global_transform.0)
                .unwrap_or_else(cgmath::Matrix4::identity);
            Some((mesh_asset.graphics_mesh_index, model_matrix))
        })
        .collect();

    let transform_count = indexed_transforms
        .iter()
        .map(|(index, _)| index + 1)
        .max()
        .unwrap_or(0);

    let mut transforms = vec![cgmath::Matrix4::identity(); transform_count];
    for (index, model_matrix) in indexed_transforms {
        transforms[index] = model_matrix;
    }

    transforms
}

pub unsafe fn rebuild_acceleration_structures(
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    graphics: &GraphicsResources,
    raytracing: &mut RayTracingData,
    waters: &[(cgmath::Matrix4<f32>, f32, f32)],
    mesh_transforms: &[cgmath::Matrix4<f32>],
) -> Result<()> {
    log!("Rebuilding acceleration structures...");

    let mut acceleration_structure = RRAccelerationStructure::new();

    // Collect vertex_buffers in the same order as BLAS creation
    let vertex_buffers: Vec<_> = graphics
        .meshes
        .iter()
        .filter(|mesh| mesh.render_to_gbuffer)
        .map(|mesh| {
            (
                &mesh.vertex_buffer.buffer,
                mesh.vertex_data.vertices.len() as u32,
                std::mem::size_of::<vulkan_data::Vertex>() as u32,
                &mesh.index_buffer.buffer,
                mesh.vertex_data.indices.len() as u32,
            )
        })
        .collect();

    for (mesh_index, mesh) in graphics.meshes.iter().enumerate() {
        if !mesh.render_to_gbuffer {
            continue;
        }

        let mut blas = RRAccelerationStructure::create_blas(
            instance,
            device,
            command_pool.as_ref(),
            &mesh.vertex_buffer.buffer,
            mesh.vertex_data.vertices.len() as u32,
            std::mem::size_of::<vulkan_data::Vertex>() as u32,
            &mesh.index_buffer.buffer,
            mesh.vertex_data.indices.len() as u32,
        )?;

        let model = mesh_transforms
            .get(mesh_index)
            .copied()
            .unwrap_or_else(cgmath::Matrix4::identity);
        blas.transform = vk::TransformMatrixKHR {
            matrix: AffineRows3x4::from_mat4(model).rows,
        };

        acceleration_structure.blas_list.push(blas);
        log!("Created BLAS for mesh");
    }

    for (model, major, minor) in waters {
        let blas = RRAccelerationStructure::create_water_blas(
            instance,
            device,
            command_pool.as_ref(),
            model,
            *major,
            *minor,
        )?;
        acceleration_structure.water_blas.push(blas);
    }

    let tlas = RRAccelerationStructure::create_tlas(
        instance,
        device,
        command_pool.as_ref(),
        &acceleration_structure.blas_list,
        &acceleration_structure.water_blas,
    )?;
    acceleration_structure.tlas = tlas;
    log!(
        "Created TLAS with {} mesh + {} water instances",
        acceleration_structure.blas_list.len(),
        acceleration_structure.water_blas.len()
    );

    acceleration_structure.fill_hit_shading_table(instance, device, &vertex_buffers, waters)?;

    raytracing.acceleration_structure = Some(acceleration_structure);
    log!("Acceleration structures rebuilt successfully");
    Ok(())
}

pub unsafe fn rebuild_acceleration_structures_from_data(
    instance: &Instance,
    rrdevice: &RRDevice,
    data: &mut AppData,
    rrcommand_pool: &Rc<RRCommandPool>,
) -> Result<()> {
    let waters = collect_water_instances(&data.ecs_world);
    let mesh_transforms = collect_mesh_transforms(&data.ecs_world, &data.ecs_assets);
    rebuild_acceleration_structures(
        instance,
        rrdevice,
        rrcommand_pool,
        &data.graphics_resources,
        &mut data.raytracing,
        &waters,
        &mesh_transforms,
    )
}

pub(super) unsafe fn update_ray_query_descriptor(
    device: &RRDevice,
    raytracing: &mut RayTracingData,
) -> Result<()> {
    raytracing.bind_ray_query_tlas(device)?;
    if raytracing.has_valid_tlas() {
        log!("Updated ray_query_descriptor with new TLAS");
    }
    Ok(())
}

pub(super) unsafe fn update_billboard_descriptor(
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
