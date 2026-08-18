use crate::core::device::*;
use crate::core::swapchain::*;
use crate::data::*;
use crate::render::pass::RRRender;
use crate::vulkan::*;
use std::fs::File;
use std::io::Read;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::vk::PrimitiveTopology;

#[derive(Clone, Debug, Default)]
pub struct RRPipeline {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

use thyllore_model_core::mesh::{VertexFormat, VertexLayout};

/// Vertex input configuration for pipeline
pub enum VertexInputConfig {
    /// Use standard Vertex struct (position, normal, texcoord, etc.)
    Standard,
    /// Use ImGui DrawVert struct (position, uv, color)
    ImGui,
    /// Use ColorVertex struct (position, color)
    Gizmo,
    /// Use BillboardVertex struct (position, texcoord)
    Billboard,
    /// Use VertexLayout from SoA mesh system
    FromLayout(VertexLayout),
    /// Custom vertex input (bindings and attributes)
    Custom {
        bindings: Vec<vk::VertexInputBindingDescription>,
        attributes: Vec<vk::VertexInputAttributeDescription>,
    },
}

fn vertex_format_to_vk(format: VertexFormat) -> vk::Format {
    match format {
        VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
        VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
    }
}

#[derive(Clone, Copy)]
pub struct DepthTestConfig {
    pub test_enable: bool,
    pub write_enable: bool,
    pub compare_op: vk::CompareOp,
}

impl Default for DepthTestConfig {
    fn default() -> Self {
        Self {
            test_enable: true,
            write_enable: true,
            compare_op: vk::CompareOp::GREATER_OR_EQUAL,
        }
    }
}

/// Blend configuration
#[derive(Clone, Copy)]
pub struct BlendConfig {
    pub enable: bool,
    pub src_color_factor: vk::BlendFactor,
    pub dst_color_factor: vk::BlendFactor,
    pub color_op: vk::BlendOp,
    pub src_alpha_factor: vk::BlendFactor,
    pub dst_alpha_factor: vk::BlendFactor,
    pub alpha_op: vk::BlendOp,
}

impl Default for BlendConfig {
    fn default() -> Self {
        Self {
            enable: true,
            src_color_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_op: vk::BlendOp::ADD,
            src_alpha_factor: vk::BlendFactor::ONE,
            dst_alpha_factor: vk::BlendFactor::ZERO,
            alpha_op: vk::BlendOp::ADD,
        }
    }
}

#[derive(Clone, Copy)]
pub struct PushConstantConfig {
    pub stage_flags: vk::ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}

/// Pipeline builder for flexible pipeline creation
pub struct PipelineBuilder {
    vertex_shader_path: String,
    fragment_shader_path: String,
    geometry_shader_path: Option<String>,
    vertex_input: VertexInputConfig,
    topology: vk::PrimitiveTopology,
    polygon_mode: vk::PolygonMode,
    cull_mode: vk::CullModeFlags,
    depth_test: DepthTestConfig,
    blend: BlendConfig,
    push_constants: Option<PushConstantConfig>,
    dynamic_states: Vec<vk::DynamicState>,
    descriptor_layouts: Vec<vk::DescriptorSetLayout>,
    msaa_samples: vk::SampleCountFlags,
    custom_render_pass: Option<vk::RenderPass>,
    mrt_attachment_count: u32,
    no_blend_attachments: Vec<u32>,
    attachment_blend_overrides: Vec<(u32, BlendConfig)>,
}

impl PipelineBuilder {
    pub fn new(vertex_shader: &str, fragment_shader: &str) -> Self {
        Self {
            vertex_shader_path: vertex_shader.to_string(),
            fragment_shader_path: fragment_shader.to_string(),
            geometry_shader_path: None,
            vertex_input: VertexInputConfig::Standard,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            depth_test: DepthTestConfig::default(),
            blend: BlendConfig::default(),
            push_constants: None,
            dynamic_states: vec![
                vk::DynamicState::VIEWPORT,
                vk::DynamicState::SCISSOR,
                vk::DynamicState::LINE_WIDTH,
            ],
            descriptor_layouts: vec![],
            msaa_samples: vk::SampleCountFlags::empty(),
            custom_render_pass: None,
            mrt_attachment_count: 1,
            no_blend_attachments: vec![],
            attachment_blend_overrides: vec![],
        }
    }

    /// Set geometry shader stage (requires the geometry_shader device feature)
    pub fn geometry_shader(mut self, path: &str) -> Self {
        self.geometry_shader_path = Some(path.to_string());
        self
    }

    /// Override blend configuration for a single MRT attachment
    pub fn attachment_blend(mut self, attachment_index: u32, config: BlendConfig) -> Self {
        self.attachment_blend_overrides
            .push((attachment_index, config));
        self
    }

    pub fn no_blend_attachment(mut self, attachment_index: u32) -> Self {
        self.no_blend_attachments.push(attachment_index);
        self
    }

    /// Set custom render pass (for G-Buffer, etc.)
    pub fn custom_render_pass(mut self, render_pass: vk::RenderPass) -> Self {
        self.custom_render_pass = Some(render_pass);
        self
    }

    /// Set number of color attachments for MRT
    pub fn mrt_attachments(mut self, count: u32) -> Self {
        self.mrt_attachment_count = count;
        self
    }

    /// Set vertex input configuration
    pub fn vertex_input(mut self, config: VertexInputConfig) -> Self {
        self.vertex_input = config;
        self
    }

    /// Set primitive topology (TRIANGLE_LIST, LINE_LIST, etc.)
    pub fn topology(mut self, topology: vk::PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Set polygon mode (FILL, LINE, POINT)
    pub fn polygon_mode(mut self, mode: vk::PolygonMode) -> Self {
        self.polygon_mode = mode;
        self
    }

    /// Set cull mode (NONE, FRONT, BACK, FRONT_AND_BACK)
    pub fn cull_mode(mut self, mode: vk::CullModeFlags) -> Self {
        self.cull_mode = mode;
        self
    }

    /// Set depth test configuration
    pub fn depth_test(mut self, config: DepthTestConfig) -> Self {
        self.depth_test = config;
        self
    }

    /// Disable depth test
    pub fn no_depth_test(mut self) -> Self {
        self.depth_test.test_enable = false;
        self.depth_test.write_enable = false;
        self
    }

    /// Set blend configuration
    pub fn blend(mut self, config: BlendConfig) -> Self {
        self.blend = config;
        self
    }

    /// Set push constant configuration
    pub fn push_constants(mut self, config: PushConstantConfig) -> Self {
        self.push_constants = Some(config);
        self
    }

    /// Set dynamic states
    pub fn dynamic_states(mut self, states: Vec<vk::DynamicState>) -> Self {
        self.dynamic_states = states;
        self
    }

    /// Set descriptor set layouts
    pub fn descriptor_layouts(mut self, layouts: Vec<vk::DescriptorSetLayout>) -> Self {
        self.descriptor_layouts = layouts;
        self
    }

    /// Set MSAA samples
    pub fn msaa_samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.msaa_samples = samples;
        self
    }

    /// Build the pipeline
    pub unsafe fn build(
        self,
        rrdevice: &RRDevice,
        rrrender: &RRRender,
        swapchain_extent: Option<vk::Extent2D>,
    ) -> Result<RRPipeline> {
        let mut rrpipeline = RRPipeline::default();

        // Load shaders
        let vert_shader_module = load_shader_module(rrdevice, &self.vertex_shader_path)?;
        let frag_shader_module = load_shader_module(rrdevice, &self.fragment_shader_path)?;
        let geom_shader_module = match &self.geometry_shader_path {
            Some(path) => Some(load_shader_module(rrdevice, path)?),
            None => None,
        };

        // Shader stages
        let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(b"main\0");
        let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(b"main\0");
        let mut shader_stages = vec![vert_stage];
        if let Some(module) = geom_shader_module {
            shader_stages.push(
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::GEOMETRY)
                    .module(module)
                    .name(b"main\0"),
            );
        }
        shader_stages.push(frag_stage);

        let (binding_descriptions, attribute_descriptions) =
            resolve_vertex_input(self.vertex_input);

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        // Input assembly
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(self.topology)
            .primitive_restart_enable(false);

        // Viewport and scissor
        let viewport = if let Some(extent) = swapchain_extent {
            vk::Viewport::builder()
                .x(0.0)
                .y(0.0)
                .width(extent.width as f32)
                .height(extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0)
                .build()
        } else {
            vk::Viewport::default()
        };

        let scissor = if let Some(extent) = swapchain_extent {
            vk::Rect2D::builder()
                .offset(vk::Offset2D { x: 0, y: 0 })
                .extent(extent)
                .build()
        } else {
            vk::Rect2D::default()
        };

        let viewports = [viewport];
        let scissors = [scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        // Rasterization state
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(self.polygon_mode)
            .line_width(1.0)
            .cull_mode(self.cull_mode)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        // Multisample state
        let msaa_samples = if !self.msaa_samples.is_empty() {
            self.msaa_samples
        } else {
            rrdevice.msaa_samples
        };
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(msaa_samples != vk::SampleCountFlags::_1)
            .min_sample_shading(if msaa_samples != vk::SampleCountFlags::_1 {
                0.9
            } else {
                1.0
            })
            .rasterization_samples(msaa_samples);

        // Depth stencil state
        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(self.depth_test.test_enable)
            .depth_write_enable(self.depth_test.write_enable)
            .depth_compare_op(self.depth_test.compare_op)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false);

        let color_blend_attachments: Vec<_> = (0..self.mrt_attachment_count)
            .map(|i| {
                let blend = self
                    .attachment_blend_overrides
                    .iter()
                    .find(|(index, _)| *index == i)
                    .map(|(_, config)| *config)
                    .unwrap_or(self.blend);
                let blend_enable = blend.enable && !self.no_blend_attachments.contains(&i);
                vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(vk::ColorComponentFlags::all())
                    .blend_enable(blend_enable)
                    .src_color_blend_factor(blend.src_color_factor)
                    .dst_color_blend_factor(blend.dst_color_factor)
                    .color_blend_op(blend.color_op)
                    .src_alpha_blend_factor(blend.src_alpha_factor)
                    .dst_alpha_blend_factor(blend.dst_alpha_factor)
                    .alpha_blend_op(blend.alpha_op)
                    .build()
            })
            .collect();
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachments);

        // Dynamic state
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&self.dynamic_states);

        // Pipeline layout
        let push_constant_ranges: Vec<vk::PushConstantRange> = self
            .push_constants
            .map(|pc| {
                vec![vk::PushConstantRange::builder()
                    .stage_flags(pc.stage_flags)
                    .offset(pc.offset)
                    .size(pc.size)
                    .build()]
            })
            .unwrap_or_default();

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&self.descriptor_layouts)
            .push_constant_ranges(&push_constant_ranges);

        rrpipeline.pipeline_layout = rrdevice
            .device
            .create_pipeline_layout(&pipeline_layout_info, None)?;

        // Create graphics pipeline
        let render_pass = self.custom_render_pass.unwrap_or(rrrender.render_pass);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(rrpipeline.pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipelines = rrdevice
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None)
            .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {:?}", e))?;

        rrpipeline.pipeline = pipelines.0[0];

        // Clean up shader modules
        rrdevice
            .device
            .destroy_shader_module(vert_shader_module, None);
        rrdevice
            .device
            .destroy_shader_module(frag_shader_module, None);
        for stage in &shader_stages {
            if stage.stage == vk::ShaderStageFlags::GEOMETRY {
                rrdevice.device.destroy_shader_module(stage.module, None);
            }
        }

        println!("Pipeline created successfully");
        Ok(rrpipeline)
    }
}

impl RRPipeline {
    /// Create a pipeline with GraphicsResources layouts (Set 0, 1, 2)
    pub unsafe fn new_with_graphics_resources(
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
        rrrender: &RRRender,
        layouts: &[vk::DescriptorSetLayout; 3],
        vertex_shader_path: &str,
        fragment_shader_path: &str,
        topology: PrimitiveTopology,
        polygon_mode: vk::PolygonMode,
        cull_mode: vk::CullModeFlags,
    ) -> Result<Self> {
        let mut builder = PipelineBuilder::new(vertex_shader_path, fragment_shader_path)
            .vertex_input(VertexInputConfig::Standard)
            .topology(topology)
            .polygon_mode(polygon_mode)
            .cull_mode(cull_mode)
            .descriptor_layouts(layouts.to_vec());

        if topology == vk::PrimitiveTopology::LINE_LIST {
            builder = builder.depth_test(DepthTestConfig {
                test_enable: true,
                write_enable: false,
                compare_op: vk::CompareOp::GREATER_OR_EQUAL,
            });
        }

        builder.build(rrdevice, rrrender, Some(rrswapchain.swapchain_extent))
    }

    /// Create a pipeline with GraphicsResources layouts (Set 0, Set 2 only - no material)
    pub unsafe fn new_with_frame_object_layouts(
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
        rrrender: &RRRender,
        layouts: &[vk::DescriptorSetLayout; 2],
        vertex_shader_path: &str,
        fragment_shader_path: &str,
        topology: PrimitiveTopology,
        polygon_mode: vk::PolygonMode,
        cull_mode: vk::CullModeFlags,
    ) -> Result<Self> {
        let mut builder = PipelineBuilder::new(vertex_shader_path, fragment_shader_path)
            .vertex_input(VertexInputConfig::Standard)
            .topology(topology)
            .polygon_mode(polygon_mode)
            .cull_mode(cull_mode)
            .descriptor_layouts(layouts.to_vec());

        if topology == vk::PrimitiveTopology::LINE_LIST {
            builder = builder.depth_test(DepthTestConfig {
                test_enable: true,
                write_enable: false,
                compare_op: vk::CompareOp::GREATER_OR_EQUAL,
            });
        }

        builder.build(rrdevice, rrrender, Some(rrswapchain.swapchain_extent))
    }

    /// Create ImGui rendering pipeline (backward compatibility)
    pub unsafe fn new_imgui(
        rrdevice: &RRDevice,
        rrrender: &RRRender,
        descriptor_set_layout: vk::DescriptorSetLayout,
        vertex_shader_path: &str,
        fragment_shader_path: &str,
        msaa_samples: vk::SampleCountFlags,
    ) -> Result<Self> {
        PipelineBuilder::new(vertex_shader_path, fragment_shader_path)
            .vertex_input(VertexInputConfig::ImGui)
            .no_depth_test()
            .push_constants(PushConstantConfig {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset: 0,
                size: std::mem::size_of::<[f32; 4]>() as u32,
            })
            .dynamic_states(vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .descriptor_layouts(vec![descriptor_set_layout])
            .msaa_samples(msaa_samples)
            .build(rrdevice, rrrender, None)
    }

    /// Create Billboard rendering pipeline with depth test
    pub unsafe fn new_billboard(
        rrdevice: &RRDevice,
        rrrender: &RRRender,
        rrswapchain: &RRSwapchain,
        descriptor_set_layout: vk::DescriptorSetLayout,
        vertex_shader_path: &str,
        fragment_shader_path: &str,
    ) -> Result<Self> {
        PipelineBuilder::new(vertex_shader_path, fragment_shader_path)
            .vertex_input(VertexInputConfig::Billboard)
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .polygon_mode(vk::PolygonMode::FILL)
            .blend(BlendConfig::default())
            .descriptor_layouts(vec![descriptor_set_layout])
            .build(rrdevice, rrrender, Some(rrswapchain.swapchain_extent))
    }

    /// Create compute pipeline for ray query or other compute shaders
    pub unsafe fn new_compute(
        rrdevice: &RRDevice,
        compute_shader_path: &str,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
    ) -> Result<Self> {
        Self::new_compute_with_push_constants(
            rrdevice,
            compute_shader_path,
            descriptor_set_layouts,
            &[],
        )
    }

    /// Create compute pipeline with push constants
    pub unsafe fn new_compute_with_push_constants(
        rrdevice: &RRDevice,
        compute_shader_path: &str,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self> {
        let device = &rrdevice.device;

        // Load compute shader
        let comp_shader_module = load_shader_module(rrdevice, compute_shader_path)?;

        // Create shader stage
        let comp_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(comp_shader_module)
            .name(b"main\0")
            .build();

        // Create pipeline layout
        let mut layout_info =
            vk::PipelineLayoutCreateInfo::builder().set_layouts(descriptor_set_layouts);
        if !push_constant_ranges.is_empty() {
            layout_info = layout_info.push_constant_ranges(push_constant_ranges);
        }
        let pipeline_layout = device.create_pipeline_layout(&layout_info.build(), None)?;

        // Create compute pipeline
        let compute_pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(comp_stage)
            .layout(pipeline_layout)
            .build();

        let pipeline = device
            .create_compute_pipelines(vk::PipelineCache::null(), &[compute_pipeline_info], None)?
            .0[0];

        // Clean up shader module
        device.destroy_shader_module(comp_shader_module, None);

        Ok(Self {
            pipeline_layout,
            pipeline,
        })
    }

    pub unsafe fn destroy(&self, device: &vulkanalia::Device) {
        if self.pipeline != vk::Pipeline::null() {
            device.destroy_pipeline(self.pipeline, None);
        }

        if self.pipeline_layout != vk::PipelineLayout::null() {
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

/// Load a shader module from file path
unsafe fn load_shader_module(rrdevice: &RRDevice, path: &str) -> Result<vk::ShaderModule> {
    let mut file = File::open(path)?;
    let mut bytecode = Vec::new();
    file.read_to_end(&mut bytecode)?;
    create_shader_module(rrdevice, &bytecode)
}

/// Create shader module from bytecode
unsafe fn create_shader_module(rrdevice: &RRDevice, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode =
        Bytecode::new(bytecode).map_err(|e| anyhow::anyhow!("Invalid shader bytecode: {:?}", e))?;
    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    Ok(rrdevice.device.create_shader_module(&info, None)?)
}

type VertexInputDescriptions = (
    Vec<vk::VertexInputBindingDescription>,
    Vec<vk::VertexInputAttributeDescription>,
);

fn resolve_vertex_input(config: VertexInputConfig) -> VertexInputDescriptions {
    match config {
        VertexInputConfig::Standard => {
            let bindings = vec![crate::data::vertex_binding_description()];
            let attributes = crate::data::vertex_attribute_descriptions().to_vec();
            (bindings, attributes)
        }
        VertexInputConfig::ImGui => resolve_imgui_vertex_input(),
        VertexInputConfig::Gizmo => resolve_simple_vertex_input(
            std::mem::size_of::<[f32; 6]>() as u32,
            &[
                (vk::Format::R32G32B32_SFLOAT, 0),
                (
                    vk::Format::R32G32B32_SFLOAT,
                    std::mem::size_of::<[f32; 3]>() as u32,
                ),
            ],
        ),
        VertexInputConfig::Billboard => resolve_simple_vertex_input(
            std::mem::size_of::<[f32; 5]>() as u32,
            &[
                (vk::Format::R32G32B32_SFLOAT, 0),
                (
                    vk::Format::R32G32_SFLOAT,
                    std::mem::size_of::<[f32; 3]>() as u32,
                ),
            ],
        ),
        VertexInputConfig::FromLayout(layout) => resolve_layout_vertex_input(&layout),
        VertexInputConfig::Custom {
            bindings,
            attributes,
        } => (bindings, attributes),
    }
}

fn resolve_imgui_vertex_input() -> VertexInputDescriptions {
    let bindings = vec![vk::VertexInputBindingDescription::builder()
        .binding(0)
        .stride(std::mem::size_of::<imgui::DrawVert>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)
        .build()];
    let attributes = vec![
        vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0)
            .build(),
        vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(8)
            .build(),
        vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R8G8B8A8_UNORM)
            .offset(16)
            .build(),
    ];
    (bindings, attributes)
}

fn resolve_simple_vertex_input(
    stride: u32,
    attrs: &[(vk::Format, u32)],
) -> VertexInputDescriptions {
    let bindings = vec![vk::VertexInputBindingDescription::builder()
        .binding(0)
        .stride(stride)
        .input_rate(vk::VertexInputRate::VERTEX)
        .build()];
    let attributes = attrs
        .iter()
        .enumerate()
        .map(|(i, (format, offset))| {
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(i as u32)
                .format(*format)
                .offset(*offset)
                .build()
        })
        .collect();
    (bindings, attributes)
}

fn resolve_layout_vertex_input(layout: &VertexLayout) -> VertexInputDescriptions {
    let bindings = vec![vk::VertexInputBindingDescription::builder()
        .binding(0)
        .stride(layout.stride)
        .input_rate(vk::VertexInputRate::VERTEX)
        .build()];

    let mut attributes = Vec::new();
    let mut offset = 0u32;
    for attr in &layout.attributes {
        attributes.push(
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(attr.shader_location)
                .format(vertex_format_to_vk(attr.format))
                .offset(offset)
                .build(),
        );
        offset += attr.format.size();
    }
    (bindings, attributes)
}
