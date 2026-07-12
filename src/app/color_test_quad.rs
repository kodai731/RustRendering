use std::ffi::c_void;
use std::mem::size_of;
use std::rc::Rc;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::asset::{AssetStorage, MeshAsset};
use crate::ecs::world::{Transform, World};
use crate::math::{Vec2, Vec3, Vec4};
use crate::render::MaterialUBO;
use crate::vulkanr::buffer::{RRIndexBuffer, RRVertexBuffer};
use crate::vulkanr::command::RRCommandPool;
use crate::vulkanr::core::RRDevice;
use crate::vulkanr::data::{Vertex, VertexData};
use crate::vulkanr::resource::graphics_resource::GraphicsResources;
use crate::vulkanr::resource::MeshBuffer;
use crate::vulkanr::vulkan::Instance;
use thyllore_vulkan_core::resource::image::{
    create_image_view, create_texture_image_pixel, create_texture_sampler,
};
use thyllore_vulkan_core::resource::raytracing_data::RayTracingData;

struct TestColorQuadSpec {
    color: Vec4,
    position_x: f32,
}

pub unsafe fn spawn_color_test_quads(
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    graphics: &mut GraphicsResources,
    raytracing: &mut RayTracingData,
    world: &mut World,
    assets: &mut AssetStorage,
) -> Result<()> {
    let specs = [
        TestColorQuadSpec {
            color: Vec4::new(0.051, 0.051, 0.051, 1.0),
            position_x: -6.0,
        },
        TestColorQuadSpec {
            color: Vec4::new(0.6, 0.0, 0.0, 1.0),
            position_x: -2.0,
        },
        TestColorQuadSpec {
            color: Vec4::new(0.0, 0.4, 0.0, 1.0),
            position_x: 2.0,
        },
        TestColorQuadSpec {
            color: Vec4::new(0.0, 0.0, 1.0, 1.0),
            position_x: 6.0,
        },
    ];

    let mesh_start_index = graphics.meshes.len();

    for (i, spec) in specs.iter().enumerate() {
        let mesh_buffer =
            create_quad_mesh_buffer(instance, device, command_pool, graphics, spec.color)?;
        let material_id = create_quad_material(instance, device, graphics, &mesh_buffer, i)?;

        graphics.meshes.push(mesh_buffer);
        graphics.mesh_material_ids.push(material_id);
    }

    raytracing.build_acceleration_structures(instance, device, command_pool, &graphics.meshes)?;

    for (i, spec) in specs.iter().enumerate() {
        let mesh_idx = mesh_start_index + i;
        let mesh = &graphics.meshes[mesh_idx];
        let label = format!("color_test_{}", i);

        let mesh_asset = MeshAsset {
            id: 0,
            name: label.clone(),
            graphics_mesh_index: mesh_idx,
            object_index: mesh.object_index,
            material_id: graphics.mesh_material_ids.get(mesh_idx).copied(),
            skeleton_id: None,
            node_index: None,
            render_to_gbuffer: true,
        };
        let asset_id = assets.add_mesh(mesh_asset);

        let mut transform = Transform::default();
        transform.translation.x = spec.position_x;

        world
            .entity()
            .with_name(&label)
            .with_transform(transform)
            .with_visible(true)
            .with_mesh(asset_id, mesh.object_index)
            .build();
    }

    log!("Spawned {} color test quads", specs.len());
    Ok(())
}

fn build_quad_vertex_data(color: Vec4) -> VertexData {
    let half = 1.5;
    let normal = Vec3::new(0.0, 0.0, 1.0);

    let vertices = vec![
        Vertex {
            pos: Vec3::new(-half, -half, 0.0),
            color,
            tex_coord: Vec2::new(0.0, 1.0),
            normal,
        },
        Vertex {
            pos: Vec3::new(half, -half, 0.0),
            color,
            tex_coord: Vec2::new(1.0, 1.0),
            normal,
        },
        Vertex {
            pos: Vec3::new(half, half, 0.0),
            color,
            tex_coord: Vec2::new(1.0, 0.0),
            normal,
        },
        Vertex {
            pos: Vec3::new(-half, half, 0.0),
            color,
            tex_coord: Vec2::new(0.0, 0.0),
            normal,
        },
    ];

    let indices = vec![0, 1, 2, 0, 2, 3];

    VertexData { vertices, indices }
}

unsafe fn create_quad_mesh_buffer(
    instance: &Instance,
    device: &RRDevice,
    command_pool: &Rc<RRCommandPool>,
    graphics: &mut GraphicsResources,
    color: Vec4,
) -> Result<MeshBuffer> {
    let vertex_data = build_quad_vertex_data(color);

    let vertex_buffer = RRVertexBuffer::new(
        instance,
        device,
        command_pool,
        (size_of::<Vertex>() * vertex_data.vertices.len()) as vk::DeviceSize,
        vertex_data.vertices.as_ptr() as *const c_void,
        vertex_data.vertices.len(),
    )?;

    let index_buffer = RRIndexBuffer::new(
        instance,
        device,
        command_pool,
        (size_of::<u32>() * vertex_data.indices.len()) as u64,
        vertex_data.indices.as_ptr() as *const c_void,
        vertex_data.indices.len(),
    )?;

    let white_pixel = vec![255u8, 255, 255, 255];
    let (image, image_memory, mip_level) =
        create_texture_image_pixel(instance, device, command_pool, &white_pixel, 1, 1)?;

    let image_view = create_image_view(
        device,
        image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
        mip_level,
    )?;
    let sampler = create_texture_sampler(device, mip_level)?;

    let object_index = graphics.objects.allocate_slot();

    Ok(MeshBuffer {
        vertex_buffer,
        index_buffer,
        vertex_data,
        image,
        image_memory,
        mip_level,
        image_view,
        sampler,
        render_to_gbuffer: true,
        object_index,
        skin_data: None,
        skeleton_id: None,
        node_index: None,
        base_vertices: Vec::new(),
        base_colors: None,
    })
}

unsafe fn create_quad_material(
    instance: &Instance,
    device: &RRDevice,
    graphics: &mut GraphicsResources,
    mesh: &MeshBuffer,
    index: usize,
) -> Result<crate::vulkanr::resource::graphics_resource::MaterialId> {
    let name = format!("color_test_material_{}", index);
    let material_id = graphics.materials.create_material_with_texture(
        instance,
        device,
        &name,
        mesh.image_view,
        mesh.sampler,
        MaterialUBO::default(),
    )?;
    Ok(material_id)
}
