use std::fs::File;
use std::io::Read;
use super::data::*;
use super::descriptor::*;
use super::device::*;
use super::swapchain::*;
use super::vulkan::*;
use crate::vulkanr::render::RRRender;
use vulkanalia::bytecode::Bytecode;

#[derive(Clone, Debug, Default)]
pub struct RRPipeline {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl RRPipeline {
    pub unsafe fn new(
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
        rrrender: &RRRender,
        rrdescriptor_set: &RRDescriptorSet,
        vertex_shader_path: &str,
        fragment_shader_path: &str,
    ) -> Self {
        let mut rrpipeline = RRPipeline::default();
        let _ = create_pipeline(
            rrdevice,
            rrswapchain,
            rrrender,
            rrdescriptor_set,
            &mut rrpipeline,
            vertex_shader_path,
            fragment_shader_path,
        );
        rrpipeline
    }
}
unsafe fn create_pipeline(
    rrdevice: &RRDevice,
    rrswapchain: &RRSwapchain,
    rrrender: &RRRender,
    rrdescriptor_set: &RRDescriptorSet,
    rrpipeline: &mut RRPipeline,
    vertex_shader_path: &str,
    fragment_shader_path: &str,
) -> Result<()> {
    let mut vert_file = File::open(vertex_shader_path)?;
    let mut frag_file = File::open(fragment_shader_path)?;
    let mut vert = Vec::new();
    let mut frag = Vec::new();
    vert_file.read_to_end(&mut vert)?;
    frag_file.read_to_end(&mut frag)?;
    let vert_shader_module = create_shader_module(rrdevice, &vert[..])?;
    let frag_shader_module = create_shader_module(rrdevice, &frag[..])?;

    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_shader_module)
        .name(b"main\0");
    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(b"main\0");

    let binding_discriptions = &[Vertex::binding_description()];
    let attribute_descriptions = Vertex::attribute_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(binding_discriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(rrswapchain.swapchain_extent.width as f32)
        .height(rrswapchain.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(rrswapchain.swapchain_extent);

    let viewports = &[viewport];
    let scissors = &[scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(viewports)
        .scissors(scissors);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(true) // https://registry.khronos.org/vulkan/specs/1.0/html/vkspec.html#primsrast-sampleshading
        .min_sample_shading(0.9) //  Minimum fraction for sample shading; closer to one is smoother.
        .sample_shading_enable(false)
        .rasterization_samples(rrdevice.msaa_samples);

    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    let attachments = &[attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    // NOTE: This will cause the configuration of these values to be ignored and you will be required to specify the data at drawing time.
    let dynamic_state = &[vk::DynamicState::VIEWPORT, vk::DynamicState::LINE_WIDTH];

    let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(dynamic_state);

    let descriptor_set_layouts = &[rrdescriptor_set.descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(descriptor_set_layouts);
    rrpipeline.pipeline_layout = rrdevice.device.create_pipeline_layout(&layout_info, None)?;

    let stages = &[vert_stage, frag_stage];

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .stencil_test_enable(false);

    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .layout(rrpipeline.pipeline_layout)
        .render_pass(rrrender.render_pass)
        .subpass(0);

    rrpipeline.pipeline = rrdevice
        .device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
        .0[0];

    rrdevice
        .device
        .destroy_shader_module(vert_shader_module, None);
    rrdevice
        .device
        .destroy_shader_module(frag_shader_module, None);
    Ok(())
}

unsafe fn create_shader_module(rrdevice: &RRDevice, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode = Bytecode::new(bytecode).unwrap();
    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    Ok(rrdevice.device.create_shader_module(&info, None)?)
}
