use std::ffi::CStr;

use crate::app::App;
use crate::ecs::systems::{
    build_water_debug_record, current_unix_time, water_debug_caustic_accum_path,
    water_debug_screenshot_path, write_water_debug_dump, WaterDebugRenderInfo,
};
use crate::vulkanr::context::{CommandState, SwapchainState};
use crate::vulkanr::data::Vertex;
use crate::vulkanr::vulkan::*;
use thyllore_math_core::write_npy_u32;

use serde_json::{json, Value};
use thyllore_vulkan_core::raytracing::RRAccelerationStructure;
use thyllore_vulkan_core::resource::mesh_buffer::MeshBuffer;

const VERTEX_PROBE_COUNT: usize = 64;

#[derive(Clone, Copy, Debug, Default)]
pub struct WaterCausticAccumStats {
    pub nonzero_count: u64,
    pub max_value: u32,
}

impl App {
    pub unsafe fn save_water_caustic_accum_npy(
        &self,
        path: &std::path::Path,
    ) -> Result<WaterCausticAccumStats> {
        let device = &self.rrdevice.device;
        let water_targets = self
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::WaterRenderTargets>()
            .ok_or_else(|| anyhow::anyhow!("water buffer not initialized"))?;
        let water_buffer = &water_targets.buffer;

        let caustic_image = water_buffer.caustic_accum_image;
        let width = water_buffer.width;
        let height = water_buffer.height;
        let image_size = (width * height * 4) as vk::DeviceSize;

        let (buffer, buffer_memory) = self.copy_image_to_buffer(
            caustic_image,
            width,
            height,
            image_size,
            vk::ImageLayout::GENERAL,
        )?;

        let data_ptr =
            device.map_memory(buffer_memory, 0, image_size, vk::MemoryMapFlags::empty())?;
        let slice = std::slice::from_raw_parts(data_ptr as *const u8, image_size as usize);

        let mut u32_data: Vec<u32> = Vec::with_capacity((width * height) as usize);
        let mut stats = WaterCausticAccumStats::default();
        for chunk in slice.chunks_exact(4) {
            let value = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if value != 0 {
                stats.nonzero_count += 1;
            }
            stats.max_value = stats.max_value.max(value);
            u32_data.push(value);
        }

        device.unmap_memory(buffer_memory);
        device.free_memory(buffer_memory, None);
        device.destroy_buffer(buffer, None);

        if let Some(directory) = path.parent() {
            std::fs::create_dir_all(directory)?;
        }
        write_npy_u32(path, &[height as usize, width as usize], &u32_data)?;

        Ok(stats)
    }

    pub fn dump_water_debug(&self) {
        self.dump_water_debug_at(self.frame % crate::app::init::MAX_FRAMES_IN_FLIGHT);
    }

    pub fn dump_water_debug_at(&self, image_index: usize) {
        let unix_time = current_unix_time();
        let mut render_info = self.collect_water_debug_render_info();

        let screenshot_path = water_debug_screenshot_path(unix_time);
        match unsafe { self.save_screenshot_to(image_index, &screenshot_path) } {
            Ok(_) => render_info.screenshot_path = Some(screenshot_path.display().to_string()),
            Err(error) => log_warn!("water debug screenshot failed: {:?}", error),
        }

        let caustic_accum_path = water_debug_caustic_accum_path(unix_time);
        match unsafe { self.save_water_caustic_accum_npy(&caustic_accum_path) } {
            Ok(stats) => {
                render_info.caustic_accum_path = Some(caustic_accum_path.display().to_string());
                render_info.caustic_accum_nonzero = Some(stats.nonzero_count);
                render_info.caustic_accum_max = Some(stats.max_value);
            }
            Err(error) => log_warn!("water debug caustic accum save failed: {:?}", error),
        }

        let record = build_water_debug_record(&self.data.ecs_world, &render_info, unix_time);
        match write_water_debug_dump(&record, unix_time) {
            Ok(path) => msg_info!("Water debug dumped: {}", path.display()),
            Err(error) => log_error!("water debug dump failed: {}", error),
        }
    }

    fn collect_water_debug_render_info(&self) -> WaterDebugRenderInfo {
        let properties = unsafe {
            self.instance
                .get_physical_device_properties(self.rrdevice.physical_device)
        };
        let gpu_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .to_string();

        let swapchain_extent = self.resource::<SwapchainState>().swapchain.swapchain_extent;
        let viewport = &self.data.viewport;
        let water_buffer_size = self
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::WaterRenderTargets>()
            .map(|targets| [targets.buffer.width, targets.buffer.height]);
        let acceleration = self.data.raytracing.acceleration_structure.as_ref();

        WaterDebugRenderInfo {
            gpu_name,
            driver_version: format_vulkan_version(properties.driver_version),
            api_version: format_vulkan_version(properties.api_version),
            swapchain_size: [swapchain_extent.width, swapchain_extent.height],
            hdr_buffer_size: viewport.hdr_buffer.as_ref().map(|b| [b.width, b.height]),
            water_buffer_size,
            mesh_count: self.data.graphics_resources.meshes.len(),
            mesh_blas_count: acceleration.map(|a| a.blas_list.len()).unwrap_or(0),
            water_blas_count: acceleration.map(|a| a.water_blas.len()).unwrap_or(0),
            hit_shading_table_capacity: acceleration
                .and_then(|a| a.hit_shading_table.as_ref())
                .map(|table| table.capacity),
            screenshot_path: None,
            caustic_accum_path: None,
            caustic_accum_nonzero: None,
            caustic_accum_max: None,
            tlas_instances: acceleration.map(|a| self.build_tlas_instances_json(a)),
            mesh_vertex_probe: acceleration.map(|_| self.build_mesh_vertex_probe_json()),
        }
    }

    fn build_tlas_instances_json(&self, acceleration: &RRAccelerationStructure) -> Value {
        let mut instances: Vec<Value> = acceleration
            .blas_list
            .iter()
            .enumerate()
            .zip(self.collect_gbuffer_mesh_indices())
            .map(|((blas_index, blas), mesh_index)| {
                let mesh = &self.data.graphics_resources.meshes[mesh_index];
                json!({
                    "blas_index": blas_index,
                    "mesh_index": mesh_index,
                    "transform": blas.transform.matrix,
                    "vertex_count": mesh.vertex_data.vertices.len(),
                    "index_count": mesh.vertex_data.indices.len(),
                })
            })
            .collect();

        let blas_count = acceleration.blas_list.len();
        instances.extend(
            acceleration
                .water_blas
                .iter()
                .enumerate()
                .map(|(water_index, blas)| {
                    json!({
                        "blas_index": blas_count + water_index,
                        "kind": "water",
                        "transform": blas.transform.matrix,
                    })
                }),
        );

        Value::Array(instances)
    }

    fn build_mesh_vertex_probe_json(&self) -> Value {
        let probes: Vec<Value> = self
            .collect_gbuffer_mesh_indices()
            .into_iter()
            .map(|mesh_index| {
                let mesh = &self.data.graphics_resources.meshes[mesh_index];
                let sample_count = mesh.vertex_data.vertices.len().min(VERTEX_PROBE_COUNT);
                let cpu_positions: Vec<[f32; 3]> = mesh.vertex_data.vertices[..sample_count]
                    .iter()
                    .map(|vertex| [vertex.pos[0], vertex.pos[1], vertex.pos[2]])
                    .collect();
                let gpu_positions =
                    match unsafe { self.read_gpu_vertex_positions(mesh, sample_count) } {
                        Ok(positions) => positions,
                        Err(error) => {
                            log_warn!("water debug vertex probe failed: {:?}", error);
                            Vec::new()
                        }
                    };

                json!({
                    "mesh_index": mesh_index,
                    "sample_count": sample_count,
                    "gpu_pos_first": gpu_positions.first(),
                    "gpu_pos_centroid64": average_position(&gpu_positions),
                    "cpu_pos_first": cpu_positions.first(),
                    "cpu_pos_centroid64": average_position(&cpu_positions),
                })
            })
            .collect();

        Value::Array(probes)
    }

    unsafe fn read_gpu_vertex_positions(
        &self,
        mesh: &MeshBuffer,
        sample_count: usize,
    ) -> Result<Vec<[f32; 3]>> {
        if sample_count == 0 {
            return Ok(Vec::new());
        }

        let device = &self.rrdevice.device;
        let stride = std::mem::size_of::<Vertex>();
        let copy_size = (stride * sample_count) as vk::DeviceSize;
        let command_pool = self.resource::<CommandState>().pool.command_pool;

        let (buffer, buffer_memory) = self.allocate_vertex_probe_buffer(copy_size)?;
        let command_buffer = self.record_and_submit_vertex_copy(
            command_pool,
            mesh.vertex_buffer.buffer,
            buffer,
            copy_size,
        )?;

        let mapped = device.map_memory(buffer_memory, 0, copy_size, vk::MemoryMapFlags::empty())?;
        let bytes = std::slice::from_raw_parts(mapped as *const u8, copy_size as usize);
        let pos_offset = std::mem::offset_of!(Vertex, pos);
        let positions = (0..sample_count)
            .map(|index| read_position(bytes, index * stride + pos_offset))
            .collect();

        device.unmap_memory(buffer_memory);
        device.free_command_buffers(command_pool, &[command_buffer]);
        device.free_memory(buffer_memory, None);
        device.destroy_buffer(buffer, None);

        Ok(positions)
    }

    unsafe fn allocate_vertex_probe_buffer(
        &self,
        size: vk::DeviceSize,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let device = &self.rrdevice.device;

        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = device.create_buffer(&buffer_info, None)?;

        let requirements = device.get_buffer_memory_requirements(buffer);
        let memory_type_index = self.get_memory_type_index(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let buffer_memory = device.allocate_memory(&allocate_info, None)?;
        device.bind_buffer_memory(buffer, buffer_memory, 0)?;

        Ok((buffer, buffer_memory))
    }

    unsafe fn record_and_submit_vertex_copy(
        &self,
        command_pool: vk::CommandPool,
        source: vk::Buffer,
        destination: vk::Buffer,
        size: vk::DeviceSize,
    ) -> Result<vk::CommandBuffer> {
        let device = &self.rrdevice.device;

        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = device.allocate_command_buffers(&allocate_info)?[0];

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device.begin_command_buffer(command_buffer, &begin_info)?;
        let region = vk::BufferCopy::builder()
            .src_offset(0)
            .dst_offset(0)
            .size(size);
        device.cmd_copy_buffer(command_buffer, source, destination, &[region]);
        device.end_command_buffer(command_buffer)?;

        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);
        device.queue_submit(
            self.rrdevice.graphics_queue,
            &[submit_info.build()],
            vk::Fence::null(),
        )?;
        device.queue_wait_idle(self.rrdevice.graphics_queue)?;

        Ok(command_buffer)
    }

    fn collect_gbuffer_mesh_indices(&self) -> Vec<usize> {
        self.data
            .graphics_resources
            .meshes
            .iter()
            .enumerate()
            .filter(|(_, mesh)| mesh.render_to_gbuffer)
            .map(|(mesh_index, _)| mesh_index)
            .collect()
    }
}

fn read_position(bytes: &[u8], offset: usize) -> [f32; 3] {
    let read_float = |component: usize| {
        let start = offset + component * std::mem::size_of::<f32>();
        let mut raw = [0u8; 4];
        raw.copy_from_slice(&bytes[start..start + 4]);
        f32::from_le_bytes(raw)
    };
    [read_float(0), read_float(1), read_float(2)]
}

fn average_position(positions: &[[f32; 3]]) -> Option<[f32; 3]> {
    if positions.is_empty() {
        return None;
    }
    let inverse_count = 1.0 / positions.len() as f32;
    let sum = positions.iter().fold([0.0f32; 3], |mut sum, position| {
        sum[0] += position[0];
        sum[1] += position[1];
        sum[2] += position[2];
        sum
    });
    Some([
        sum[0] * inverse_count,
        sum[1] * inverse_count,
        sum[2] * inverse_count,
    ])
}

fn format_vulkan_version(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::version_major(version),
        vk::version_minor(version),
        vk::version_patch(version)
    )
}
