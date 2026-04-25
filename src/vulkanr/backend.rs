use std::ffi::c_void;
use std::mem::size_of;
use std::rc::Rc;

use anyhow::Result;
use cgmath::{Matrix4, Vector3, Vector4};
use vulkanalia::prelude::v1_0::*;

use crate::ecs::component::LineMesh;
use crate::ecs::resource::billboard::BillboardData;
use crate::ecs::resource::DistanceAttenuation;
use crate::ecs::resource::ProjectionData;
use crate::render::{
    BufferMemoryType, FrameUBO, IndexBufferHandle, MeshId, ObjectUBO, RenderBackend,
    VertexBufferHandle,
};
use crate::vulkanr::command::RRCommandPool;
use crate::vulkanr::core::device::RRDevice;
use crate::vulkanr::data::Vertex;
use crate::vulkanr::image::RRImage;
use crate::vulkanr::resource::graphics_resource::GraphicsResources;
use crate::vulkanr::resource::GpuBufferRegistry;
use crate::vulkanr::vulkan::Instance;
use thyllore_vulkan_core::data::SceneUniformData;
use thyllore_vulkan_core::raytracing::RRAccelerationStructure;
use thyllore_vulkan_core::resource::raytracing_data::RayTracingData;

pub struct VulkanBackend<'a> {
    pub instance: &'a Instance,
    pub device: &'a RRDevice,
    pub command_pool: Rc<RRCommandPool>,
    pub graphics: &'a mut GraphicsResources,
    pub raytracing: &'a mut RayTracingData,
    pub buffer_registry: &'a mut GpuBufferRegistry,
}

impl<'a> VulkanBackend<'a> {
    pub fn new(
        instance: &'a Instance,
        device: &'a RRDevice,
        command_pool: Rc<RRCommandPool>,
        graphics: &'a mut GraphicsResources,
        raytracing: &'a mut RayTracingData,
        buffer_registry: &'a mut GpuBufferRegistry,
    ) -> Self {
        Self {
            instance,
            device,
            command_pool,
            graphics,
            raytracing,
            buffer_registry,
        }
    }

    fn acceleration_structure(&mut self) -> &mut Option<RRAccelerationStructure> {
        &mut self.raytracing.acceleration_structure
    }
}

impl<'a> RenderBackend for VulkanBackend<'a> {
    unsafe fn upload_mesh_vertices(&mut self, mesh_id: MeshId) -> Result<()> {
        if mesh_id >= self.graphics.meshes.len() {
            return Ok(());
        }

        let mesh = &mut self.graphics.meshes[mesh_id];
        let vertices = &mesh.vertex_data.vertices;
        let vertex_count = vertices.len();
        let vertex_stride = size_of::<Vertex>();

        mesh.vertex_buffer.update(
            self.instance,
            self.device,
            self.command_pool.as_ref(),
            (vertex_stride * vertex_count) as vk::DeviceSize,
            vertices.as_ptr() as *const c_void,
            vertex_count,
        )?;

        Ok(())
    }

    unsafe fn update_acceleration_structure(&mut self, mesh_ids: &[MeshId]) -> Result<()> {
        let Some(ref accel_struct) = self.raytracing.acceleration_structure else {
            return Ok(());
        };

        for &mesh_id in mesh_ids {
            if mesh_id >= self.graphics.meshes.len() {
                continue;
            }
            if mesh_id >= accel_struct.blas_list.len() {
                continue;
            }

            let mesh = &self.graphics.meshes[mesh_id];
            let blas = &accel_struct.blas_list[mesh_id];

            RRAccelerationStructure::update_blas(
                self.instance,
                self.device,
                self.command_pool.as_ref(),
                blas,
                &mesh.vertex_buffer.buffer,
                mesh.vertex_data.vertices.len() as u32,
                size_of::<Vertex>() as u32,
                &mesh.index_buffer.buffer,
                mesh.vertex_data.indices.len() as u32,
            )?;
        }

        Ok(())
    }

    unsafe fn rebuild_tlas(&mut self) -> Result<()> {
        let Some(ref accel_struct) = self.raytracing.acceleration_structure else {
            return Ok(());
        };

        let tlas = &accel_struct.tlas;
        RRAccelerationStructure::update_tlas(
            self.instance,
            self.device,
            self.command_pool.as_ref(),
            tlas,
            &accel_struct.blas_list,
        )?;

        Ok(())
    }

    unsafe fn create_gizmo_buffers(
        &mut self,
        mesh: &mut LineMesh,
        memory_type: BufferMemoryType,
    ) -> Result<()> {
        let vertex_handle = if memory_type == BufferMemoryType::DeviceLocal {
            self.buffer_registry.create_vertex_buffer(
                self.instance,
                self.device,
                self.command_pool.as_ref(),
                &mesh.vertices,
                BufferMemoryType::DeviceLocal,
            )?
        } else {
            self.buffer_registry.create_host_visible_vertex_buffer(
                self.instance,
                self.device,
                &mesh.vertices,
                0,
            )?
        };
        mesh.vertex_buffer_handle = vertex_handle;

        let index_handle = self.buffer_registry.create_index_buffer(
            self.instance,
            self.device,
            self.command_pool.as_ref(),
            &mesh.indices,
        )?;
        mesh.index_buffer_handle = index_handle;

        Ok(())
    }

    unsafe fn update_gizmo_vertex_buffer(&self, mesh: &LineMesh) -> Result<()> {
        self.buffer_registry.update_vertex_buffer(
            self.device,
            mesh.vertex_buffer_handle,
            &mesh.vertices,
        )?;
        Ok(())
    }

    unsafe fn destroy_gizmo_buffers(&mut self, mesh: &mut LineMesh) {
        self.buffer_registry
            .destroy_vertex_buffer(self.device, mesh.vertex_buffer_handle);
        self.buffer_registry
            .destroy_index_buffer(self.device, mesh.index_buffer_handle);
        mesh.vertex_buffer_handle = VertexBufferHandle::INVALID;
        mesh.index_buffer_handle = IndexBufferHandle::INVALID;
    }

    unsafe fn update_or_create_line_buffers(&mut self, mesh: &mut LineMesh) -> Result<()> {
        if mesh.vertices.is_empty() {
            return Ok(());
        }

        let vertex_data_size = (std::mem::size_of_val(mesh.vertices.as_slice())) as u64;

        if !mesh.vertex_buffer_handle.is_valid()
            || self
                .buffer_registry
                .get_vertex_buffer_size(mesh.vertex_buffer_handle)
                < vertex_data_size
        {
            if mesh.vertex_buffer_handle.is_valid() {
                self.buffer_registry
                    .destroy_vertex_buffer(self.device, mesh.vertex_buffer_handle);
            }
            let vertex_handle = self.buffer_registry.create_host_visible_vertex_buffer(
                self.instance,
                self.device,
                &mesh.vertices,
                1024,
            )?;
            mesh.vertex_buffer_handle = vertex_handle;
        } else {
            self.buffer_registry.update_vertex_buffer(
                self.device,
                mesh.vertex_buffer_handle,
                &mesh.vertices,
            )?;
        }

        let index_data_size = (std::mem::size_of::<u32>() * mesh.indices.len()) as u64;

        if !mesh.index_buffer_handle.is_valid()
            || self
                .buffer_registry
                .get_index_buffer_size(mesh.index_buffer_handle)
                < index_data_size
        {
            if mesh.index_buffer_handle.is_valid() {
                self.buffer_registry
                    .destroy_index_buffer(self.device, mesh.index_buffer_handle);
            }
            let index_handle = self.buffer_registry.create_host_visible_index_buffer(
                self.instance,
                self.device,
                &mesh.indices,
            )?;
            mesh.index_buffer_handle = index_handle;
        } else {
            self.buffer_registry.update_index_buffer(
                self.device,
                mesh.index_buffer_handle,
                &mesh.indices,
            )?;
        }

        Ok(())
    }

    unsafe fn destroy_line_buffers(&mut self, mesh: &mut LineMesh) {
        self.buffer_registry
            .destroy_vertex_buffer(self.device, mesh.vertex_buffer_handle);
        self.buffer_registry
            .destroy_index_buffer(self.device, mesh.index_buffer_handle);
        mesh.vertex_buffer_handle = VertexBufferHandle::INVALID;
        mesh.index_buffer_handle = IndexBufferHandle::INVALID;
    }

    unsafe fn create_billboard_buffers(&mut self, billboard: &mut BillboardData) -> Result<()> {
        billboard.mesh.vertex_buffer_handle =
            self.buffer_registry.create_host_visible_vertex_buffer(
                self.instance,
                self.device,
                &billboard.mesh.vertices,
                256,
            )?;

        billboard.mesh.index_buffer_handle =
            self.buffer_registry.create_host_visible_index_buffer(
                self.instance,
                self.device,
                &billboard.mesh.indices,
            )?;

        let white_pixel: [u8; 4] = [255, 255, 255, 255];
        billboard.render_state.texture = Some(
            RRImage::new_from_pixels(
                self.instance,
                self.device,
                &self.command_pool,
                &white_pixel,
                1,
                1,
            )
            .map_err(|e| anyhow::anyhow!("Failed to create billboard texture: {}", e))?,
        );

        Ok(())
    }

    unsafe fn update_frame_ubo(
        &mut self,
        proj_data: &ProjectionData,
        camera_pos: Vector3<f32>,
        light_pos: Vector3<f32>,
        light_color: Vector3<f32>,
        image_index: usize,
    ) -> Result<()> {
        let ubo = FrameUBO {
            view: proj_data.view,
            proj: proj_data.proj,
            camera_pos: Vector4::new(camera_pos.x, camera_pos.y, camera_pos.z, 1.0),
            light_pos: Vector4::new(light_pos.x, light_pos.y, light_pos.z, 1.0),
            light_color: Vector4::new(light_color.x, light_color.y, light_color.z, 1.0),
        };

        self.graphics
            .frame_set
            .update(self.device, image_index, &ubo)?;

        Ok(())
    }

    unsafe fn update_object_ubo(
        &mut self,
        model_matrix: Matrix4<f32>,
        object_index: usize,
        image_index: usize,
    ) -> Result<()> {
        let ubo = ObjectUBO {
            model: model_matrix,
        };
        self.graphics
            .objects
            .update(self.device, image_index, object_index, &ubo)?;
        Ok(())
    }

    unsafe fn update_scene_uniform(
        &mut self,
        view: Matrix4<f32>,
        proj: Matrix4<f32>,
        light_pos: Vector3<f32>,
        light_color: Vector3<f32>,
        debug_mode: i32,
        shadow_strength: f32,
        distance_attenuation: DistanceAttenuation,
        exposure_value: f32,
    ) -> Result<()> {
        let (scene_buffer, scene_memory) = match (
            self.raytracing.scene_uniform_buffer,
            self.raytracing.scene_uniform_buffer_memory,
        ) {
            (Some(b), Some(m)) => (b, m),
            _ => return Ok(()),
        };

        let scene_data = SceneUniformData {
            light_position: crate::math::Vec4::new(light_pos.x, light_pos.y, light_pos.z, 1.0),
            light_color: crate::math::Vec4::new(light_color.x, light_color.y, light_color.z, 1.0),
            view,
            proj,
            debug_mode,
            shadow_strength,
            enable_distance_attenuation: distance_attenuation.as_int(),
            exposure_value,
        };

        let data_ptr = self.device.device.map_memory(
            scene_memory,
            0,
            std::mem::size_of::<SceneUniformData>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        std::ptr::copy_nonoverlapping(
            &scene_data as *const SceneUniformData,
            data_ptr as *mut SceneUniformData,
            1,
        );

        self.device.device.unmap_memory(scene_memory);

        Ok(())
    }

    unsafe fn update_billboard_ubo(
        &mut self,
        billboard: &mut BillboardData,
        model: Matrix4<f32>,
        view: Matrix4<f32>,
        proj: Matrix4<f32>,
        image_index: usize,
    ) -> Result<()> {
        use crate::vulkanr::data::UniformBufferObject;

        for i in 0..billboard.render_state.descriptor_set.rrdata.len() {
            let rrdata = &mut billboard.render_state.descriptor_set.rrdata[i];

            let ubo = UniformBufferObject { model, view, proj };
            rrdata.rruniform_buffers[image_index].update(self.device, &ubo)?;
        }

        Ok(())
    }
}
