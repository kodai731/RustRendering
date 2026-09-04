use crate::core::device::*;
use crate::descriptor::{PassShaders, ShaderStage};
use crate::resource::buffer::create_buffer;
use crate::vulkan::*;
use std::fs::File;
use std::io::Read;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::vk::{KhrGetPhysicalDeviceProperties2Extension, KhrRayTracingPipelineExtension};

#[derive(Clone, Debug)]
pub struct RRRayTracingPipeline {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub sbt_buffer: vk::Buffer,
    pub sbt_memory: vk::DeviceMemory,
    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub miss_region: vk::StridedDeviceAddressRegionKHR,
    pub hit_region: vk::StridedDeviceAddressRegionKHR,
    pub callable_region: vk::StridedDeviceAddressRegionKHR,
}

impl RRRayTracingPipeline {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        pass: &PassShaders,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self> {
        let device = &rrdevice.device;

        // Load shader modules for all 5 stages
        let raygen_shader = pass
            .stage(ShaderStage::RayGeneration)
            .ok_or_else(|| anyhow::anyhow!("pass `{}` has no RayGeneration stage", pass.name()))?;
        let miss_shader = pass
            .stage(ShaderStage::Miss)
            .ok_or_else(|| anyhow::anyhow!("pass `{}` has no Miss stage", pass.name()))?;
        let intersection_shader = pass
            .stage(ShaderStage::Intersection)
            .ok_or_else(|| anyhow::anyhow!("pass `{}` has no Intersection stage", pass.name()))?;

        // Find both closest-hit stages by file name
        let procedural_closest_hit = pass
            .stages
            .iter()
            .find(|s| s.path.contains("waterTorusRchit"))
            .ok_or_else(|| {
                anyhow::anyhow!("pass `{}` has no waterTorusRchit stage", pass.name())
            })?;
        let triangle_closest_hit = pass
            .stages
            .iter()
            .find(|s| s.path.contains("sceneRchit"))
            .ok_or_else(|| anyhow::anyhow!("pass `{}` has no sceneRchit stage", pass.name()))?;

        let raygen_module = load_shader_module(rrdevice, raygen_shader.path)?;
        let miss_module = load_shader_module(rrdevice, miss_shader.path)?;
        let intersection_module = load_shader_module(rrdevice, intersection_shader.path)?;
        let procedural_closest_hit_module =
            load_shader_module(rrdevice, procedural_closest_hit.path)?;
        let triangle_closest_hit_module = load_shader_module(rrdevice, triangle_closest_hit.path)?;

        // Create shader stages
        let raygen_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
            .module(raygen_module)
            .name(b"main\0")
            .build();

        let miss_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .module(miss_module)
            .name(b"main\0")
            .build();

        let intersection_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::INTERSECTION_KHR)
            .module(intersection_module)
            .name(b"main\0")
            .build();

        let procedural_closest_hit_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(procedural_closest_hit_module)
            .name(b"main\0")
            .build();

        let triangle_closest_hit_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(triangle_closest_hit_module)
            .name(b"main\0")
            .build();

        let stages: [vk::PipelineShaderStageCreateInfo; 5] = [
            raygen_stage,
            miss_stage,
            intersection_stage,
            procedural_closest_hit_stage,
            triangle_closest_hit_stage,
        ];
        // Create ray tracing shader groups
        let groups: [vk::RayTracingShaderGroupCreateInfoKHR; 4] = [
            vk::RayTracingShaderGroupCreateInfoKHR::builder()
                .type_(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .build(),
            vk::RayTracingShaderGroupCreateInfoKHR::builder()
                .type_(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(1)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .build(),
            vk::RayTracingShaderGroupCreateInfoKHR::builder()
                .type_(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(2)
                .closest_hit_shader(3)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .build(),
            vk::RayTracingShaderGroupCreateInfoKHR::builder()
                .type_(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(4)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .build(),
        ];

        // Create pipeline layout
        let mut layout_info =
            vk::PipelineLayoutCreateInfo::builder().set_layouts(descriptor_set_layouts);
        if !push_constant_ranges.is_empty() {
            layout_info = layout_info.push_constant_ranges(push_constant_ranges);
        }
        let pipeline_layout = device.create_pipeline_layout(&layout_info.build(), None)?;

        // Create ray tracing pipeline
        let rt_pipeline_info = vk::RayTracingPipelineCreateInfoKHR::builder()
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(1)
            .layout(pipeline_layout)
            .build();

        let pipelines = device.create_ray_tracing_pipelines_khr(
            vk::DeferredOperationKHR::null(),
            vk::PipelineCache::null(),
            &[rt_pipeline_info],
            None,
        )?;
        let pipeline = pipelines.0[0];

        // Destroy shader modules
        device.destroy_shader_module(raygen_module, None);
        device.destroy_shader_module(miss_module, None);
        device.destroy_shader_module(intersection_module, None);
        device.destroy_shader_module(procedural_closest_hit_module, None);
        device.destroy_shader_module(triangle_closest_hit_module, None);

        // Fetch physical device ray tracing properties
        let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut props2 = vk::PhysicalDeviceProperties2::builder().push_next(&mut rt_props);
        instance.get_physical_device_properties2(rrdevice.physical_device, &mut props2);

        let handle_size: u64 = rt_props.shader_group_handle_size as u64;
        let handle_alignment: u64 = rt_props.shader_group_handle_alignment as u64;
        let base_alignment: u64 = rt_props.shader_group_base_alignment as u64;

        anyhow::ensure!(
            handle_size > 0 && handle_alignment > 0 && base_alignment > 0,
            "ray tracing pipeline properties not available"
        );

        let handle_stride = align_up(handle_size, handle_alignment);
        let region_size = align_up(handle_stride, base_alignment);
        let hit_region_size = align_up(2 * handle_stride, base_alignment);
        let sbt_size = 2 * region_size + hit_region_size;

        // Allocate SBT buffer
        let (sbt_buffer, sbt_memory) = create_buffer(
            instance,
            rrdevice,
            sbt_size,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Get shader group handles (4 groups)
        let handle_count = 4 * handle_size as usize;
        let mut handles = vec![0u8; handle_count];
        device.get_ray_tracing_shader_group_handles_khr(pipeline, 0, 4, &mut handles);

        // Map buffer and copy handles to each region
        // Layout: [raygen (group 0)] [miss (group 1)] [hit record 0 = triangle group (group 3)] [hit record 1 = procedural group (group 2)]
        let mapped_ptr = device.map_memory(sbt_memory, 0, sbt_size, vk::MemoryMapFlags::empty())?;
        let slice = std::slice::from_raw_parts_mut(mapped_ptr as *mut u8, sbt_size as usize);

        // Raygen handle (group 0) at offset 0
        {
            let dst = &mut slice[0..handle_size as usize];
            let src = &handles[0..handle_size as usize];
            dst.copy_from_slice(src);
        }
        // Miss handle (group 1) at offset region_size
        {
            let dst = &mut slice[region_size as usize..region_size as usize + handle_size as usize];
            let src = &handles[handle_size as usize..2 * handle_size as usize];
            dst.copy_from_slice(src);
        }
        // Triangle group handle (group 3) at offset 2*region_size (hit record 0)
        {
            let dst =
                &mut slice[(2 * region_size) as usize..(2 * region_size + handle_size) as usize];
            let src = &handles[3 * handle_size as usize..4 * handle_size as usize];
            dst.copy_from_slice(src);
        }
        // Procedural group handle (group 2) at offset 2*region_size + handle_stride (hit record 1)
        {
            let dst = &mut slice[(2 * region_size + handle_stride) as usize
                ..(2 * region_size + handle_stride + handle_size) as usize];
            let src = &handles[2 * handle_size as usize..3 * handle_size as usize];
            dst.copy_from_slice(src);
        }
        device.unmap_memory(sbt_memory);

        // Get buffer device address
        let base_address = device
            .get_buffer_device_address(&vk::BufferDeviceAddressInfo::builder().buffer(sbt_buffer));

        // Set up SBT regions
        let raygen_region = vk::StridedDeviceAddressRegionKHR::builder()
            .device_address(base_address + 0 * region_size)
            .size(region_size)
            .stride(region_size)
            .build();

        let miss_region = vk::StridedDeviceAddressRegionKHR::builder()
            .device_address(base_address + 1 * region_size)
            .size(region_size)
            .stride(handle_stride)
            .build();

        let hit_region = vk::StridedDeviceAddressRegionKHR::builder()
            .device_address(base_address + 2 * region_size)
            .size(hit_region_size)
            .stride(handle_stride)
            .build();

        let callable_region = vk::StridedDeviceAddressRegionKHR::builder()
            .device_address(base_address + 2 * region_size + hit_region_size)
            .size(0)
            .stride(handle_stride)
            .build();
        Ok(Self {
            pipeline_layout,
            pipeline,
            sbt_buffer,
            sbt_memory,
            raygen_region,
            miss_region,
            hit_region,
            callable_region,
        })
    }

    pub unsafe fn destroy(&self, device: &vulkanalia::Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_buffer(self.sbt_buffer, None);
        device.free_memory(self.sbt_memory, None);
    }
}
unsafe fn load_shader_module(rrdevice: &RRDevice, path: &str) -> Result<vk::ShaderModule> {
    let mut file = File::open(path)?;
    let mut bytecode = Vec::new();
    file.read_to_end(&mut bytecode)?;
    create_shader_module(rrdevice, &bytecode)
}

unsafe fn create_shader_module(rrdevice: &RRDevice, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode =
        Bytecode::new(bytecode).map_err(|e| anyhow::anyhow!("Invalid shader bytecode: {:?}", e))?;
    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    Ok(rrdevice.device.create_shader_module(&info, None)?)
}

pub fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) & !(a - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up_basic() {
        assert_eq!(align_up(17, 16), 32);
    }

    #[test]
    fn test_align_up_already_aligned() {
        assert_eq!(align_up(32, 16), 32);
    }

    #[test]
    fn test_align_up_zero() {
        assert_eq!(align_up(0, 64), 0);
    }
}
