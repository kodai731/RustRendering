use serde::Serialize;

use crate::app::effect_hooks::EffectHooks;
use crate::app::post_process::PostProcessFrameTargets;
use crate::app::viewport::ViewportState;
use crate::asset::AssetStorage;
use crate::ecs::World;
use crate::platform::ImguiData;
use crate::vulkanr::renderer::onion_skin_buffers::OnionSkinGpuState;
use crate::vulkanr::resource::graphics_resource::GraphicsResources;
use crate::vulkanr::resource::{GpuBufferRegistry, PipelineStorage};
use thyllore_vulkan_core::resource::raytracing_data::RayTracingData;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum LightMoveTarget {
    None,
    XMin,
    XMax,
    YMin,
    YMax,
    ZMin,
    ZMax,
}

#[derive(Debug, Default)]
pub struct AppData {
    pub graphics_resources: GraphicsResources,
    pub imgui: ImguiData,
    pub raytracing: RayTracingData,
    pub ecs_world: World,
    pub ecs_assets: AssetStorage,
    pub buffer_registry: GpuBufferRegistry,
    pub pipeline_storage: PipelineStorage,
    pub viewport: ViewportState,
    pub effect_hooks: EffectHooks,
    pub post_process: PostProcessFrameTargets,
    pub onion_skin_gpu: Option<OnionSkinGpuState>,
}
