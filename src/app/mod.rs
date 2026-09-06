pub mod cleanup;
pub mod color_test_quad;
mod command_recording;
pub mod data;
pub mod effect_render_targets;
pub mod frame_context;
pub mod init;
pub mod model_loader;
pub mod post_process_targets;
pub mod render;
pub mod render_context;
pub mod scene_model;
pub mod update;
pub mod util;
pub mod viewport;
pub mod water_frame_targets;

pub use frame_context::FrameContext;
pub use render_context::RenderContext;

pub use data::AppData;
pub use init::*;

use crate::ecs::{ResMut, ResRef, Resource};
use crate::vulkanr::device::*;

use std::time::Instant;
use vulkanalia::prelude::v1_0::*;

pub struct App {
    pub entry: Entry,
    pub instance: Instance,
    pub rrdevice: RRDevice,
    pub data: AppData,
    pub frame: usize,
    pub resized: bool,
    pub start: Instant,
    pub last_update_time: f32,
    pub gpu_timestamp_profiler: thyllore_vulkan_core::GpuTimestampProfiler,
    pub last_frame_instant: Option<Instant>,
}

impl App {
    pub fn resource<R: Resource>(&self) -> ResRef<R> {
        self.data.ecs_world.resource::<R>()
    }

    pub fn resource_mut<R: Resource>(&self) -> ResMut<R> {
        self.data.ecs_world.resource_mut::<R>()
    }

    pub fn get_resource<R: Resource>(&self) -> Option<ResRef<R>> {
        self.data.ecs_world.get_resource::<R>()
    }

    pub fn pipeline_storage(&self) -> &crate::vulkanr::resource::PipelineStorage {
        &self.data.pipeline_storage
    }
}
