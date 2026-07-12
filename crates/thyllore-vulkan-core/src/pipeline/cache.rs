use anyhow::Result;
use std::collections::HashMap;
use vulkanalia::prelude::v1_0::*;

use crate::core::device::RRDevice;
use crate::pipeline::builder::{PipelineBuilder, RRPipeline};
use crate::render::pass::RRRender;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PipelineKey {
    pub vertex_shader: String,
    pub fragment_shader: String,
    pub topology: u32,
    pub polygon_mode: u32,
    pub cull_mode: u32,
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub blend_enable: bool,
    pub msaa_samples: u32,
    pub mrt_attachment_count: u32,
    pub render_pass_id: u64,
}

impl PipelineKey {
    pub fn new(
        vertex_shader: &str,
        fragment_shader: &str,
        topology: vk::PrimitiveTopology,
        polygon_mode: vk::PolygonMode,
        cull_mode: vk::CullModeFlags,
        depth_test_enable: bool,
        depth_write_enable: bool,
        blend_enable: bool,
        msaa_samples: vk::SampleCountFlags,
        mrt_attachment_count: u32,
        render_pass: vk::RenderPass,
    ) -> Self {
        Self {
            vertex_shader: vertex_shader.to_string(),
            fragment_shader: fragment_shader.to_string(),
            topology: topology.as_raw() as u32,
            polygon_mode: polygon_mode.as_raw() as u32,
            cull_mode: cull_mode.bits(),
            depth_test_enable,
            depth_write_enable,
            blend_enable,
            msaa_samples: msaa_samples.bits(),
            mrt_attachment_count,
            render_pass_id: render_pass.as_raw(),
        }
    }
}

#[derive(Default)]
pub struct PipelineCache {
    cache: HashMap<PipelineKey, RRPipeline>,
    vulkan_cache: vk::PipelineCache,
}

impl PipelineCache {
    pub unsafe fn new(rrdevice: &RRDevice) -> Result<Self> {
        let cache_info = vk::PipelineCacheCreateInfo::builder();
        let vulkan_cache = rrdevice.device.create_pipeline_cache(&cache_info, None)?;

        Ok(Self {
            cache: HashMap::new(),
            vulkan_cache,
        })
    }

    pub fn get(&self, key: &PipelineKey) -> Option<&RRPipeline> {
        self.cache.get(key)
    }

    pub fn insert(&mut self, key: PipelineKey, pipeline: RRPipeline) {
        self.cache.insert(key, pipeline);
    }

    pub fn contains(&self, key: &PipelineKey) -> bool {
        self.cache.contains_key(key)
    }

    pub fn vulkan_cache(&self) -> vk::PipelineCache {
        self.vulkan_cache
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        for (_, pipeline) in self.cache.drain() {
            pipeline.destroy(device);
        }

        if self.vulkan_cache != vk::PipelineCache::null() {
            device.destroy_pipeline_cache(self.vulkan_cache, None);
            self.vulkan_cache = vk::PipelineCache::null();
        }
    }
}

pub struct PipelineManager {
    cache: PipelineCache,
}

impl PipelineManager {
    pub unsafe fn new(rrdevice: &RRDevice) -> Result<Self> {
        let cache = PipelineCache::new(rrdevice)?;
        Ok(Self { cache })
    }

    pub unsafe fn get_or_create(
        &mut self,
        rrdevice: &RRDevice,
        rrrender: &RRRender,
        builder: PipelineBuilder,
        key: PipelineKey,
        swapchain_extent: Option<vk::Extent2D>,
    ) -> Result<&RRPipeline> {
        if !self.cache.contains(&key) {
            let pipeline = builder.build(rrdevice, rrrender, swapchain_extent)?;
            self.cache.insert(key.clone(), pipeline);
        }

        Ok(self
            .cache
            .get(&key)
            .expect("pipeline was just inserted into cache above"))
    }

    pub fn get(&self, key: &PipelineKey) -> Option<&RRPipeline> {
        self.cache.get(key)
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        self.cache.destroy(device);
    }
}
