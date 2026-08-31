use crate::core::device::*;
use crate::descriptor::{PassShaders, ShaderStage};
use crate::vulkan::*;
use std::fs::File;
use std::io::Read;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::vk::KhrRayTracingPipelineExtension;

#[derive(Clone, Debug, Default)]
pub struct RRRayTracingPipeline {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl RRRayTracingPipeline {
    pub unsafe fn new(
        rrdevice: &RRDevice,
        pass: &PassShaders,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self> {
        let device = &rrdevice.device;

        // Load shader modules for all 4 stages
        let raygen_shader = pass
            .stage(ShaderStage::RayGeneration)
            .ok_or_else(|| anyhow::anyhow!("pass `{}` has no RayGeneration stage", pass.name()))?;
        let miss_shader = pass
            .stage(ShaderStage::Miss)
            .ok_or_else(|| anyhow::anyhow!("pass `{}` has no Miss stage", pass.name()))?;
        let intersection_shader = pass
            .stage(ShaderStage::Intersection)
            .ok_or_else(|| anyhow::anyhow!("pass `{}` has no Intersection stage", pass.name()))?;
        let closest_hit_shader = pass
            .stage(ShaderStage::ClosestHit)
            .ok_or_else(|| anyhow::anyhow!("pass `{}` has no ClosestHit stage", pass.name()))?;

        let raygen_module = load_shader_module(rrdevice, raygen_shader.path)?;
        let miss_module = load_shader_module(rrdevice, miss_shader.path)?;
        let intersection_module = load_shader_module(rrdevice, intersection_shader.path)?;
        let closest_hit_module = load_shader_module(rrdevice, closest_hit_shader.path)?;

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

        let closest_hit_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(closest_hit_module)
            .name(b"main\0")
            .build();

        let stages: [vk::PipelineShaderStageCreateInfo; 4] = [
            raygen_stage,
            miss_stage,
            intersection_stage,
            closest_hit_stage,
        ];

        // Create ray tracing shader groups
        let groups: [vk::RayTracingShaderGroupCreateInfoKHR; 3] = [
            vk::RayTracingShaderGroupCreateInfoKHR::builder()
                .type_(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .build(),
            vk::RayTracingShaderGroupCreateInfoKHR::builder()
                .type_(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(1)
                .build(),
            vk::RayTracingShaderGroupCreateInfoKHR::builder()
                .type_(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                .intersection_shader(2)
                .closest_hit_shader(3)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
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
        device.destroy_shader_module(closest_hit_module, None);

        Ok(Self {
            pipeline_layout,
            pipeline,
        })
    }

    pub unsafe fn destroy(&self, device: &vulkanalia::Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
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
