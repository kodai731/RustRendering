use anyhow::Result;
use std::rc::Rc;
use vulkanalia::prelude::v1_0::*;

use crate::command::RRCommandPool;
use crate::core::device::RRDevice;
use crate::core::swapchain::RRSwapchain;
use crate::data::{self as vulkan_data, SceneUniformData};
use crate::descriptor::{
    RRAutoExposureAverageDescriptorSet, RRAutoExposureHistogramDescriptorSet,
    RRBillboardDescriptorSet, RRBloomDescriptorSets, RRCompositeDescriptorSet, RRDofDescriptorSet,
    RRRayQueryDescriptorSet, RRToneMapDescriptorSet,
};
use crate::pipeline::{
    BlendConfig, DepthTestConfig, PipelineBuilder, PushConstantConfig, RRPipeline,
    VertexInputConfig,
};
use crate::raytracing::RRAccelerationStructure;
use crate::render::RRRender;
use crate::renderer::push_constants::{GBufferPushConstants, OnionSkinPushConstants};
use crate::resource::buffer::create_buffer;
use crate::resource::graphics_resource::{GraphicsResources, MeshBuffer};
use crate::resource::image::{create_nearest_sampler, create_texture_sampler};
use crate::resource::{BloomChain, OnionSkinPassResources, RRGBuffer};

#[derive(Clone, Debug, Default)]
pub struct RayTracingData {
    pub gbuffer: Option<RRGBuffer>,
    pub gbuffer_pipeline: Option<RRPipeline>,
    pub gbuffer_sampler: Option<vk::Sampler>,
    pub object_id_sampler: Option<vk::Sampler>,

    pub acceleration_structure: Option<RRAccelerationStructure>,

    pub ray_query_pipeline: Option<RRPipeline>,
    pub ray_query_descriptor: Option<RRRayQueryDescriptorSet>,

    pub composite_pipeline: Option<RRPipeline>,
    pub composite_descriptor: Option<RRCompositeDescriptorSet>,

    pub tonemap_pipeline: Option<RRPipeline>,
    pub tonemap_descriptor: Option<RRToneMapDescriptorSet>,

    pub bloom_downsample_pipeline: Option<RRPipeline>,
    pub bloom_upsample_pipeline: Option<RRPipeline>,
    pub bloom_descriptors: Option<RRBloomDescriptorSets>,

    pub dof_pipeline: Option<RRPipeline>,
    pub dof_descriptor: Option<RRDofDescriptorSet>,

    pub auto_exposure_histogram_pipeline: Option<RRPipeline>,
    pub auto_exposure_average_pipeline: Option<RRPipeline>,
    pub auto_exposure_histogram_descriptor: Option<RRAutoExposureHistogramDescriptorSet>,
    pub auto_exposure_average_descriptor: Option<RRAutoExposureAverageDescriptorSet>,

    pub onion_skin_pass: Option<OnionSkinPassResources>,

    pub scene_uniform_buffer: Option<vk::Buffer>,
    pub scene_uniform_buffer_memory: Option<vk::DeviceMemory>,
}

impl RayTracingData {
    pub fn has_valid_tlas(&self) -> bool {
        self.acceleration_structure
            .as_ref()
            .and_then(|a| a.tlas.acceleration_structure)
            .is_some()
    }

    pub fn is_available(&self) -> bool {
        self.gbuffer.is_some()
            && self.gbuffer_pipeline.is_some()
            && self.ray_query_pipeline.is_some()
            && self.composite_pipeline.is_some()
    }

    pub unsafe fn init_gbuffer(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
        rrcommand_pool: &RRCommandPool,
    ) -> Result<()> {
        log!("init_gbuffer: starting...");
        log!(
            "init_gbuffer: swapchain extent {}x{}",
            rrswapchain.swapchain_extent.width,
            rrswapchain.swapchain_extent.height
        );

        let gbuffer = RRGBuffer::new(
            instance,
            rrdevice,
            rrswapchain.swapchain_extent.width,
            rrswapchain.swapchain_extent.height,
        )?;

        log!("init_gbuffer: RRGBuffer::new succeeded");

        if let Err(e) = gbuffer.transition_layouts(rrdevice, rrcommand_pool.command_pool) {
            log_warn!("init_gbuffer: transition_layouts failed (ignored): {:?}", e);
        }

        self.gbuffer = Some(gbuffer);

        log!(
            "init_gbuffer: completed, gbuffer is_some: {}",
            self.gbuffer.is_some()
        );
        Ok(())
    }

    pub unsafe fn build_acceleration_structures(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        rrcommand_pool: &Rc<RRCommandPool>,
        meshes: &[MeshBuffer],
    ) -> Result<()> {
        log!("Building acceleration structures...");

        let mut acceleration_structure = RRAccelerationStructure::new();

        for mesh in meshes {
            if !mesh.render_to_gbuffer {
                continue;
            }

            let blas = RRAccelerationStructure::create_blas(
                instance,
                rrdevice,
                rrcommand_pool,
                &mesh.vertex_buffer.buffer,
                mesh.vertex_data.vertices.len() as u32,
                std::mem::size_of::<vulkan_data::Vertex>() as u32,
                &mesh.index_buffer.buffer,
                mesh.vertex_data.indices.len() as u32,
            )?;

            acceleration_structure.blas_list.push(blas);
            log!("Created BLAS for mesh");
        }

        if !acceleration_structure.blas_list.is_empty() {
            let tlas = RRAccelerationStructure::create_tlas(
                instance,
                rrdevice,
                rrcommand_pool,
                &acceleration_structure.blas_list,
            )?;
            acceleration_structure.tlas = tlas;
            log!(
                "Created TLAS with {} instances",
                acceleration_structure.blas_list.len()
            );
        }

        if acceleration_structure.blas_list.is_empty() {
            self.acceleration_structure = None;
        } else {
            self.acceleration_structure = Some(acceleration_structure);
        }
        log!("Acceleration structures built successfully");
        Ok(())
    }

    pub unsafe fn create_pipelines(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        rrswapchain: &RRSwapchain,
        rrrender: &RRRender,
        graphics_resources: &GraphicsResources,
        billboard_descriptor_set: &mut RRBillboardDescriptorSet,
        offscreen_render_pass: Option<vk::RenderPass>,
        offscreen_extent: Option<vk::Extent2D>,
        hdr_render_pass: Option<vk::RenderPass>,
    ) -> Result<()> {
        let render_layouts = [
            graphics_resources.frame_set.layout,
            graphics_resources.materials.layout,
            graphics_resources.objects.layout,
        ];

        self.gbuffer_pipeline = Some(build_gbuffer_pipeline(
            rrdevice,
            rrrender,
            rrswapchain,
            &render_layouts,
        )?);

        let scene_buffer = self.init_scene_uniform_buffer(instance, rrdevice)?;

        let (ray_query_descriptor, ray_query_pipeline) = build_ray_query_pipeline(
            rrdevice,
            &self.gbuffer,
            &self.acceleration_structure,
            scene_buffer,
        )?;
        self.ray_query_pipeline = Some(ray_query_pipeline);
        self.ray_query_descriptor = Some(ray_query_descriptor);

        let gbuffer_sampler = create_texture_sampler(rrdevice, 1)?;
        self.gbuffer_sampler = Some(gbuffer_sampler);

        let object_id_sampler = create_nearest_sampler(rrdevice)?;
        self.object_id_sampler = Some(object_id_sampler);

        let (composite_descriptor, composite_pipeline) = build_composite_pipeline(
            instance,
            rrdevice,
            rrswapchain,
            rrrender,
            &self.gbuffer,
            gbuffer_sampler,
            object_id_sampler,
            scene_buffer,
            billboard_descriptor_set,
            offscreen_render_pass,
            offscreen_extent,
            hdr_render_pass,
        )?;
        self.composite_pipeline = Some(composite_pipeline);
        self.composite_descriptor = Some(composite_descriptor);

        Ok(())
    }

    unsafe fn init_scene_uniform_buffer(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
    ) -> Result<vk::Buffer> {
        let (scene_buffer, scene_memory) = create_buffer(
            instance,
            rrdevice,
            std::mem::size_of::<SceneUniformData>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        self.scene_uniform_buffer = Some(scene_buffer);
        self.scene_uniform_buffer_memory = Some(scene_memory);
        Ok(scene_buffer)
    }

    pub unsafe fn create_onion_skin_pipeline(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        rrrender: &RRRender,
        graphics_resources: &GraphicsResources,
        offscreen_resolve_image_view: vk::ImageView,
        offscreen_format: vk::Format,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let (ghost_image, ghost_image_memory, ghost_image_view, ghost_sampler) =
            OnionSkinPassResources::create_ghost_buffer(instance, rrdevice, width, height)?;

        let ghost_render_pass = OnionSkinPassResources::create_ghost_render_pass(rrdevice)?;

        let render_layouts = [
            graphics_resources.frame_set.layout,
            graphics_resources.materials.layout,
            graphics_resources.objects.layout,
        ];

        let ghost_pipeline = PipelineBuilder::new(
            "assets/shaders/gbufferVert.spv",
            "assets/shaders/onionSkinFrag.spv",
        )
        .vertex_input(VertexInputConfig::Standard)
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .custom_render_pass(ghost_render_pass)
        .mrt_attachments(1)
        .msaa_samples(vk::SampleCountFlags::_1)
        .depth_test(DepthTestConfig {
            test_enable: false,
            write_enable: false,
            compare_op: vk::CompareOp::ALWAYS,
        })
        .blend(BlendConfig {
            enable: true,
            src_color_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_op: vk::BlendOp::ADD,
            src_alpha_factor: vk::BlendFactor::SRC_ALPHA,
            dst_alpha_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            alpha_op: vk::BlendOp::ADD,
        })
        .descriptor_layouts(render_layouts.to_vec())
        .push_constants(PushConstantConfig {
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: std::mem::size_of::<OnionSkinPushConstants>() as u32,
        })
        .build(rrdevice, rrrender, Some(vk::Extent2D { width, height }))?;

        let ghost_framebuffer = OnionSkinPassResources::create_single_framebuffer(
            rrdevice,
            ghost_render_pass,
            ghost_image_view,
            width,
            height,
        )?;

        let composite_render_pass =
            OnionSkinPassResources::create_composite_render_pass(rrdevice, offscreen_format)?;

        let (composite_descriptor_layout, composite_descriptor_pool, composite_descriptor_set) =
            OnionSkinPassResources::create_composite_descriptor(
                rrdevice,
                ghost_image_view,
                ghost_sampler,
            )?;

        let composite_pipeline = PipelineBuilder::new(
            "assets/shaders/tonemapVert.spv",
            "assets/shaders/onionSkinCompositeFrag.spv",
        )
        .vertex_input(VertexInputConfig::Custom {
            bindings: vec![],
            attributes: vec![],
        })
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .polygon_mode(vk::PolygonMode::FILL)
        .no_depth_test()
        .custom_render_pass(composite_render_pass)
        .msaa_samples(vk::SampleCountFlags::_1)
        .blend(BlendConfig {
            enable: true,
            src_color_factor: vk::BlendFactor::ONE,
            dst_color_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_op: vk::BlendOp::ADD,
            src_alpha_factor: vk::BlendFactor::ONE,
            dst_alpha_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            alpha_op: vk::BlendOp::ADD,
        })
        .descriptor_layouts(vec![composite_descriptor_layout])
        .build(rrdevice, rrrender, Some(vk::Extent2D { width, height }))?;

        let composite_framebuffer = OnionSkinPassResources::create_single_framebuffer(
            rrdevice,
            composite_render_pass,
            offscreen_resolve_image_view,
            width,
            height,
        )?;

        self.onion_skin_pass = Some(OnionSkinPassResources {
            ghost_image,
            ghost_image_memory,
            ghost_image_view,
            ghost_sampler,
            ghost_render_pass,
            ghost_framebuffer,
            ghost_pipeline,
            composite_render_pass,
            composite_framebuffer,
            composite_pipeline,
            composite_descriptor_layout,
            composite_descriptor_pool,
            composite_descriptor_set,
            width,
            height,
        });

        log!("Created onion skin pass: {}x{}", width, height);
        Ok(())
    }

    pub unsafe fn create_tonemap_pipeline(
        &mut self,
        rrdevice: &RRDevice,
        rrrender: &RRRender,
        hdr_image_view: vk::ImageView,
        hdr_sampler: vk::Sampler,
        position_image_view: vk::ImageView,
        position_sampler: vk::Sampler,
        scene_buffer: vk::Buffer,
        scene_buffer_size: vk::DeviceSize,
        offscreen_render_pass: vk::RenderPass,
        offscreen_extent: vk::Extent2D,
    ) -> Result<()> {
        let mut tonemap_descriptor = RRToneMapDescriptorSet {
            descriptor_set_layout: RRToneMapDescriptorSet::create_layout(rrdevice)?,
            descriptor_pool: RRToneMapDescriptorSet::create_pool(rrdevice)?,
            descriptor_set: vk::DescriptorSet::null(),
        };

        tonemap_descriptor.allocate_and_update(
            rrdevice,
            hdr_image_view,
            hdr_sampler,
            position_image_view,
            position_sampler,
            scene_buffer,
            scene_buffer_size,
        )?;

        let tonemap_pipeline = PipelineBuilder::new(
            "assets/shaders/tonemapVert.spv",
            "assets/shaders/tonemapFrag.spv",
        )
        .vertex_input(VertexInputConfig::Custom {
            bindings: vec![],
            attributes: vec![],
        })
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .polygon_mode(vk::PolygonMode::FILL)
        .depth_test(DepthTestConfig {
            test_enable: true,
            write_enable: true,
            compare_op: vk::CompareOp::ALWAYS,
        })
        .custom_render_pass(offscreen_render_pass)
        .descriptor_layouts(vec![tonemap_descriptor.descriptor_set_layout])
        .push_constants(PushConstantConfig {
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: 24,
        })
        .build(rrdevice, rrrender, Some(offscreen_extent))?;

        self.tonemap_pipeline = Some(tonemap_pipeline);
        self.tonemap_descriptor = Some(tonemap_descriptor);

        Ok(())
    }

    pub unsafe fn create_bloom_pipelines(
        &mut self,
        rrdevice: &RRDevice,
        rrrender: &RRRender,
        hdr_image_view: vk::ImageView,
        bloom_chain: &BloomChain,
    ) -> Result<()> {
        let mip_count = bloom_chain.mip_levels.len();
        let total_sets = (mip_count + mip_count.saturating_sub(1)) as u32;

        let mut bloom_descriptors = RRBloomDescriptorSets {
            descriptor_set_layout: RRBloomDescriptorSets::create_layout(rrdevice)?,
            descriptor_pool: RRBloomDescriptorSets::create_pool(rrdevice, total_sets)?,
            downsample_sets: Vec::new(),
            upsample_sets: Vec::new(),
        };

        let mip_views: Vec<vk::ImageView> = bloom_chain
            .mip_levels
            .iter()
            .map(|m| m.image_view)
            .collect();

        bloom_descriptors.allocate_and_update(
            rrdevice,
            hdr_image_view,
            &mip_views,
            bloom_chain.sampler,
        )?;

        let downsample_pipeline = PipelineBuilder::new(
            "assets/shaders/tonemapVert.spv",
            "assets/shaders/bloomDownsampleFrag.spv",
        )
        .vertex_input(VertexInputConfig::Custom {
            bindings: vec![],
            attributes: vec![],
        })
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .polygon_mode(vk::PolygonMode::FILL)
        .no_depth_test()
        .custom_render_pass(bloom_chain.downsample_render_pass)
        .msaa_samples(vk::SampleCountFlags::_1)
        .descriptor_layouts(vec![bloom_descriptors.descriptor_set_layout])
        .push_constants(PushConstantConfig {
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: 12,
        })
        .build(rrdevice, rrrender, None)?;

        let upsample_pipeline = PipelineBuilder::new(
            "assets/shaders/tonemapVert.spv",
            "assets/shaders/bloomUpsampleFrag.spv",
        )
        .vertex_input(VertexInputConfig::Custom {
            bindings: vec![],
            attributes: vec![],
        })
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .polygon_mode(vk::PolygonMode::FILL)
        .no_depth_test()
        .custom_render_pass(bloom_chain.upsample_render_pass)
        .msaa_samples(vk::SampleCountFlags::_1)
        .blend(BlendConfig {
            enable: true,
            src_color_factor: vk::BlendFactor::ONE,
            dst_color_factor: vk::BlendFactor::ONE,
            color_op: vk::BlendOp::ADD,
            src_alpha_factor: vk::BlendFactor::ONE,
            dst_alpha_factor: vk::BlendFactor::ONE,
            alpha_op: vk::BlendOp::ADD,
        })
        .descriptor_layouts(vec![bloom_descriptors.descriptor_set_layout])
        .build(rrdevice, rrrender, None)?;

        self.bloom_downsample_pipeline = Some(downsample_pipeline);
        self.bloom_upsample_pipeline = Some(upsample_pipeline);
        self.bloom_descriptors = Some(bloom_descriptors);
        log!("Created bloom pipelines with {} mip levels", mip_count);

        Ok(())
    }

    pub unsafe fn create_dof_pipeline(
        &mut self,
        rrdevice: &RRDevice,
        rrrender: &RRRender,
        hdr_image_view: vk::ImageView,
        hdr_sampler: vk::Sampler,
        depth_image_view: vk::ImageView,
        depth_sampler: vk::Sampler,
        dof_render_pass: vk::RenderPass,
    ) -> Result<()> {
        let mut dof_descriptor = RRDofDescriptorSet {
            descriptor_set_layout: RRDofDescriptorSet::create_layout(rrdevice)?,
            descriptor_pool: RRDofDescriptorSet::create_pool(rrdevice)?,
            descriptor_set: vk::DescriptorSet::null(),
        };

        dof_descriptor.allocate_and_update(
            rrdevice,
            hdr_image_view,
            hdr_sampler,
            depth_image_view,
            depth_sampler,
        )?;

        let dof_pipeline = PipelineBuilder::new(
            "assets/shaders/tonemapVert.spv",
            "assets/shaders/dofFrag.spv",
        )
        .vertex_input(VertexInputConfig::Custom {
            bindings: vec![],
            attributes: vec![],
        })
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .polygon_mode(vk::PolygonMode::FILL)
        .no_depth_test()
        .custom_render_pass(dof_render_pass)
        .msaa_samples(vk::SampleCountFlags::_1)
        .descriptor_layouts(vec![dof_descriptor.descriptor_set_layout])
        .push_constants(PushConstantConfig {
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: 32,
        })
        .build(rrdevice, rrrender, None)?;

        self.dof_pipeline = Some(dof_pipeline);
        self.dof_descriptor = Some(dof_descriptor);
        log!("Created DOF pipeline and descriptor set");

        Ok(())
    }

    pub unsafe fn create_auto_exposure_pipelines(
        &mut self,
        rrdevice: &RRDevice,
        hdr_image_view: vk::ImageView,
        hdr_sampler: vk::Sampler,
        histogram_buffer: vk::Buffer,
        histogram_buffer_size: u64,
        luminance_buffer: vk::Buffer,
        luminance_buffer_size: u64,
    ) -> Result<()> {
        let mut histogram_descriptor = RRAutoExposureHistogramDescriptorSet {
            descriptor_set_layout: RRAutoExposureHistogramDescriptorSet::create_layout(rrdevice)?,
            descriptor_pool: RRAutoExposureHistogramDescriptorSet::create_pool(rrdevice)?,
            descriptor_set: vk::DescriptorSet::null(),
        };

        histogram_descriptor.allocate_and_update(
            rrdevice,
            hdr_image_view,
            hdr_sampler,
            histogram_buffer,
            histogram_buffer_size,
        )?;

        let histogram_push_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(12)
            .build();

        let histogram_pipeline = RRPipeline::new_compute_with_push_constants(
            rrdevice,
            "assets/shaders/autoExposureHistogram.spv",
            &[histogram_descriptor.descriptor_set_layout],
            &[histogram_push_range],
        )?;

        let mut average_descriptor = RRAutoExposureAverageDescriptorSet {
            descriptor_set_layout: RRAutoExposureAverageDescriptorSet::create_layout(rrdevice)?,
            descriptor_pool: RRAutoExposureAverageDescriptorSet::create_pool(rrdevice)?,
            descriptor_set: vk::DescriptorSet::null(),
        };

        average_descriptor.allocate_and_update(
            rrdevice,
            histogram_buffer,
            histogram_buffer_size,
            luminance_buffer,
            luminance_buffer_size,
        )?;

        let average_push_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(40)
            .build();

        let average_pipeline = RRPipeline::new_compute_with_push_constants(
            rrdevice,
            "assets/shaders/autoExposureAverage.spv",
            &[average_descriptor.descriptor_set_layout],
            &[average_push_range],
        )?;

        self.auto_exposure_histogram_pipeline = Some(histogram_pipeline);
        self.auto_exposure_average_pipeline = Some(average_pipeline);
        self.auto_exposure_histogram_descriptor = Some(histogram_descriptor);
        self.auto_exposure_average_descriptor = Some(average_descriptor);

        Ok(())
    }
}

unsafe fn build_gbuffer_pipeline(
    rrdevice: &RRDevice,
    rrrender: &RRRender,
    rrswapchain: &RRSwapchain,
    render_layouts: &[vk::DescriptorSetLayout],
) -> Result<RRPipeline> {
    PipelineBuilder::new(
        "assets/shaders/gbufferVert.spv",
        "assets/shaders/gbufferFrag.spv",
    )
    .vertex_input(VertexInputConfig::Standard)
    .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
    .polygon_mode(vk::PolygonMode::FILL)
    .custom_render_pass(rrrender.gbuffer_render_pass)
    .mrt_attachments(4)
    .no_blend_attachment(3)
    .msaa_samples(vk::SampleCountFlags::_1)
    .descriptor_layouts(render_layouts.to_vec())
    .push_constants(PushConstantConfig {
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        offset: 0,
        size: std::mem::size_of::<GBufferPushConstants>() as u32,
    })
    .build(rrdevice, rrrender, Some(rrswapchain.swapchain_extent))
}

unsafe fn build_ray_query_pipeline(
    rrdevice: &RRDevice,
    gbuffer: &Option<RRGBuffer>,
    acceleration_structure: &Option<RRAccelerationStructure>,
    scene_buffer: vk::Buffer,
) -> Result<(RRRayQueryDescriptorSet, RRPipeline)> {
    let mut descriptor = RRRayQueryDescriptorSet {
        descriptor_set_layout: RRRayQueryDescriptorSet::create_layout(rrdevice)?,
        descriptor_pool: RRRayQueryDescriptorSet::create_pool(rrdevice)?,
        descriptor_set: vk::DescriptorSet::null(),
    };

    if let (Some(gbuffer), Some(accel_struct)) = (gbuffer, acceleration_structure) {
        if let Some(tlas) = accel_struct.tlas.acceleration_structure {
            descriptor.allocate_and_update(
                rrdevice,
                gbuffer.position_image_view,
                gbuffer.normal_image_view,
                gbuffer.shadow_mask_image_view,
                tlas,
                scene_buffer,
            )?;
        }
    }

    let push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<f32>() as u32)
        .build();

    let pipeline = RRPipeline::new_compute_with_push_constants(
        rrdevice,
        "assets/shaders/rayQueryShadow.spv",
        &[descriptor.descriptor_set_layout],
        &[push_constant_range],
    )?;

    Ok((descriptor, pipeline))
}

#[allow(clippy::too_many_arguments)]
unsafe fn build_composite_pipeline(
    instance: &Instance,
    rrdevice: &RRDevice,
    rrswapchain: &RRSwapchain,
    rrrender: &RRRender,
    gbuffer: &Option<RRGBuffer>,
    gbuffer_sampler: vk::Sampler,
    object_id_sampler: vk::Sampler,
    scene_buffer: vk::Buffer,
    billboard_descriptor_set: &mut RRBillboardDescriptorSet,
    offscreen_render_pass: Option<vk::RenderPass>,
    offscreen_extent: Option<vk::Extent2D>,
    hdr_render_pass: Option<vk::RenderPass>,
) -> Result<(RRCompositeDescriptorSet, RRPipeline)> {
    let mut descriptor = RRCompositeDescriptorSet {
        descriptor_set_layout: RRCompositeDescriptorSet::create_layout(rrdevice)?,
        descriptor_pool: RRCompositeDescriptorSet::create_pool(rrdevice)?,
        descriptor_set: vk::DescriptorSet::null(),
        selection_buffer: vk::Buffer::null(),
        selection_buffer_memory: vk::DeviceMemory::null(),
    };

    if let Some(gbuffer) = gbuffer {
        descriptor.allocate_and_update(
            instance,
            rrdevice,
            gbuffer.position_image_view,
            gbuffer_sampler,
            gbuffer.normal_image_view,
            gbuffer_sampler,
            gbuffer.shadow_mask_image_view,
            gbuffer_sampler,
            gbuffer.albedo_image_view,
            gbuffer_sampler,
            scene_buffer,
            gbuffer.object_id_image_view,
            object_id_sampler,
        )?;

        billboard_descriptor_set.update_position_sampler(
            rrdevice,
            rrswapchain,
            gbuffer.position_image_view,
            gbuffer_sampler,
        )?;
    }

    let mut builder = PipelineBuilder::new(
        "assets/shaders/compositeVert.spv",
        "assets/shaders/compositeFrag.spv",
    )
    .vertex_input(VertexInputConfig::Custom {
        bindings: vec![],
        attributes: vec![],
    })
    .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
    .polygon_mode(vk::PolygonMode::FILL)
    .descriptor_layouts(vec![descriptor.descriptor_set_layout])
    .push_constants(PushConstantConfig {
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        offset: 0,
        size: 4,
    });

    if let Some(render_pass) = hdr_render_pass {
        builder = builder
            .no_depth_test()
            .custom_render_pass(render_pass)
            .msaa_samples(vk::SampleCountFlags::_1);
    } else if let Some(render_pass) = offscreen_render_pass {
        builder = builder
            .depth_test(DepthTestConfig {
                test_enable: true,
                write_enable: true,
                compare_op: vk::CompareOp::ALWAYS,
            })
            .custom_render_pass(render_pass);
    } else {
        builder = builder.depth_test(DepthTestConfig {
            test_enable: true,
            write_enable: true,
            compare_op: vk::CompareOp::ALWAYS,
        });
    }

    let extent = offscreen_extent.unwrap_or(rrswapchain.swapchain_extent);
    let pipeline = builder.build(rrdevice, rrrender, Some(extent))?;

    Ok((descriptor, pipeline))
}
