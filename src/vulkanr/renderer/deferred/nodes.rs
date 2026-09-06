use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::pass_recording::*;
use crate::app::App;
use crate::hooks::pass::{
    CoreTarget, PassGraph, PassStage, RenderPassNode, ShaderStage, TargetAccess, TargetRef,
    TargetUse,
};

const HDR_COLOR: TargetRef = TargetRef::Core(CoreTarget::HdrColor);
const OFFSCREEN: TargetRef = TargetRef::Core(CoreTarget::Offscreen);
const ONION_GHOST: TargetRef = TargetRef::Core(CoreTarget::OnionSkinGhost);
const MAX_BLOOM_MIPS: usize = 8;

const fn cleared_attachment() -> TargetAccess {
    TargetAccess::Attachment {
        initial_layout: vk::ImageLayout::UNDEFINED,
        final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    }
}

const fn loaded_attachment(layout: vk::ImageLayout) -> TargetAccess {
    TargetAccess::Attachment {
        initial_layout: layout,
        final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    }
}

fn tonemap_input(app: &App) -> TargetRef {
    match app.data.post_process.dof_output {
        Some(handle) if app.data.raytracing.dof_pipeline.is_some() => TargetRef::Transient(handle),
        _ => HDR_COLOR,
    }
}

fn bloom_mip(app: &App, mip_index: usize) -> Option<TargetRef> {
    app.data
        .post_process
        .bloom_handles
        .get(mip_index)
        .map(|handle| TargetRef::Transient(*handle))
}

fn is_bloom_enabled(app: &App) -> bool {
    app.data
        .ecs_world
        .get_resource::<crate::ecs::resource::BloomSettings>()
        .is_some_and(|settings| settings.enabled)
        && !app.data.post_process.bloom_handles.is_empty()
}

fn is_onion_skin_active(app: &App) -> bool {
    app.data.raytracing.onion_skin_pass.is_some()
        && app
            .data
            .onion_skin_gpu
            .as_ref()
            .is_some_and(|gpu| gpu.source_mesh_index.is_some() && gpu.active_ghost_count() > 0)
}

fn is_auto_exposure_enabled(app: &App) -> bool {
    app.data
        .ecs_world
        .get_resource::<crate::ecs::resource::AutoExposure>()
        .is_some_and(|settings| settings.enabled)
        && app.data.viewport.auto_exposure_buffers.is_some()
}

pub struct CompositeHdrNode;

impl RenderPassNode for CompositeHdrNode {
    fn name(&self) -> &'static str {
        "composite_hdr"
    }

    fn stage(&self) -> PassStage {
        PassStage::Lighting
    }

    fn writes(&self, _app: &App) -> Vec<TargetUse> {
        vec![TargetUse::new(HDR_COLOR, cleared_attachment())]
    }

    unsafe fn record(&self, app: &App, cmd: vk::CommandBuffer, _: usize, _: usize) -> Result<()> {
        record_composite_to_hdr(app, cmd)
    }
}

pub struct OnionSkinNode;

impl RenderPassNode for OnionSkinNode {
    fn name(&self) -> &'static str {
        "onion_skin"
    }

    fn stage(&self) -> PassStage {
        PassStage::Lighting
    }

    fn writes(&self, app: &App) -> Vec<TargetUse> {
        if !is_onion_skin_active(app) {
            return Vec::new();
        }
        vec![TargetUse::new(ONION_GHOST, cleared_attachment())]
    }

    unsafe fn record(
        &self,
        app: &App,
        cmd: vk::CommandBuffer,
        image_index: usize,
        _: usize,
    ) -> Result<()> {
        record_onion_skin_pass(app, cmd, image_index)
    }
}

pub struct BloomDownsampleNode {
    mip_index: usize,
}

impl RenderPassNode for BloomDownsampleNode {
    fn name(&self) -> &'static str {
        BLOOM_DOWNSAMPLE_NAMES[self.mip_index]
    }

    fn stage(&self) -> PassStage {
        PassStage::PostProcess
    }

    fn reads(&self, app: &App) -> Vec<TargetUse> {
        if !is_bloom_enabled(app) || bloom_mip(app, self.mip_index).is_none() {
            return Vec::new();
        }
        let source = match self.mip_index {
            0 => Some(HDR_COLOR),
            index => bloom_mip(app, index - 1),
        };
        source
            .map(|target| TargetUse::new(target, TargetAccess::Sampled(ShaderStage::Fragment)))
            .into_iter()
            .collect()
    }

    fn writes(&self, app: &App) -> Vec<TargetUse> {
        if !is_bloom_enabled(app) {
            return Vec::new();
        }
        bloom_mip(app, self.mip_index)
            .map(|target| TargetUse::new(target, cleared_attachment()))
            .into_iter()
            .collect()
    }

    unsafe fn record(
        &self,
        app: &App,
        cmd: vk::CommandBuffer,
        _: usize,
        frame_slot: usize,
    ) -> Result<()> {
        record_bloom_downsample(app, cmd, self.mip_index, frame_slot)
    }
}

pub struct BloomUpsampleNode {
    pass_index: usize,
}

impl BloomUpsampleNode {
    fn target_mip(&self, app: &App) -> Option<usize> {
        thyllore_vulkan_core::renderer::bloom_upsample_target_mip(
            app.data.post_process.bloom_handles.len(),
            self.pass_index,
        )
    }
}

impl RenderPassNode for BloomUpsampleNode {
    fn name(&self) -> &'static str {
        BLOOM_UPSAMPLE_NAMES[self.pass_index]
    }

    fn stage(&self) -> PassStage {
        PassStage::PostProcess
    }

    fn reads(&self, app: &App) -> Vec<TargetUse> {
        if !is_bloom_enabled(app) {
            return Vec::new();
        }
        self.target_mip(app)
            .and_then(|target| bloom_mip(app, target + 1))
            .map(|source| TargetUse::new(source, TargetAccess::Sampled(ShaderStage::Fragment)))
            .into_iter()
            .collect()
    }

    fn writes(&self, app: &App) -> Vec<TargetUse> {
        if !is_bloom_enabled(app) {
            return Vec::new();
        }
        self.target_mip(app)
            .and_then(|target| bloom_mip(app, target))
            .map(|target| {
                TargetUse::new(
                    target,
                    loaded_attachment(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                )
            })
            .into_iter()
            .collect()
    }

    unsafe fn record(
        &self,
        app: &App,
        cmd: vk::CommandBuffer,
        _: usize,
        frame_slot: usize,
    ) -> Result<()> {
        record_bloom_upsample(app, cmd, self.pass_index, frame_slot)
    }
}

pub struct DofNode;

impl RenderPassNode for DofNode {
    fn name(&self) -> &'static str {
        "dof"
    }

    fn stage(&self) -> PassStage {
        PassStage::PostProcess
    }

    fn reads(&self, app: &App) -> Vec<TargetUse> {
        if app.data.post_process.dof_output.is_none() || app.data.raytracing.dof_pipeline.is_none()
        {
            return Vec::new();
        }
        vec![TargetUse::new(
            HDR_COLOR,
            TargetAccess::Sampled(ShaderStage::Fragment),
        )]
    }

    fn writes(&self, app: &App) -> Vec<TargetUse> {
        if app.data.raytracing.dof_pipeline.is_none() {
            return Vec::new();
        }
        app.data
            .post_process
            .dof_output
            .map(|handle| TargetUse::new(TargetRef::Transient(handle), cleared_attachment()))
            .into_iter()
            .collect()
    }

    unsafe fn record(&self, app: &App, cmd: vk::CommandBuffer, _: usize, _: usize) -> Result<()> {
        record_dof(app, cmd)
    }
}

pub struct AutoExposureNode;

impl RenderPassNode for AutoExposureNode {
    fn name(&self) -> &'static str {
        "auto_exposure"
    }

    fn stage(&self) -> PassStage {
        PassStage::PostProcess
    }

    fn reads(&self, app: &App) -> Vec<TargetUse> {
        if !is_auto_exposure_enabled(app) {
            return Vec::new();
        }
        vec![TargetUse::new(
            tonemap_input(app),
            TargetAccess::Sampled(ShaderStage::Compute),
        )]
    }

    unsafe fn record(
        &self,
        app: &App,
        cmd: vk::CommandBuffer,
        _: usize,
        frame_slot: usize,
    ) -> Result<()> {
        record_auto_exposure(app, cmd, frame_slot)
    }
}

pub struct TonemapNode;

impl RenderPassNode for TonemapNode {
    fn name(&self) -> &'static str {
        "tonemap"
    }

    fn stage(&self) -> PassStage {
        PassStage::Final
    }

    fn reads(&self, app: &App) -> Vec<TargetUse> {
        let sampled = TargetAccess::Sampled(ShaderStage::Fragment);
        let mut reads = vec![TargetUse::new(tonemap_input(app), sampled)];
        if is_bloom_enabled(app) {
            reads.extend(bloom_mip(app, 0).map(|target| TargetUse::new(target, sampled)));
        }
        reads
    }

    fn writes(&self, _app: &App) -> Vec<TargetUse> {
        vec![TargetUse::new(OFFSCREEN, cleared_attachment())]
    }

    unsafe fn record(
        &self,
        app: &App,
        cmd: vk::CommandBuffer,
        image_index: usize,
        _: usize,
    ) -> Result<()> {
        record_tonemap_to_offscreen(app, cmd, image_index)
    }
}

pub struct OnionCompositeNode;

impl RenderPassNode for OnionCompositeNode {
    fn name(&self) -> &'static str {
        "onion_composite"
    }

    fn stage(&self) -> PassStage {
        PassStage::Final
    }

    fn reads(&self, app: &App) -> Vec<TargetUse> {
        if !is_onion_skin_active(app) {
            return Vec::new();
        }
        vec![TargetUse::new(
            ONION_GHOST,
            TargetAccess::Sampled(ShaderStage::Fragment),
        )]
    }

    fn writes(&self, app: &App) -> Vec<TargetUse> {
        if !is_onion_skin_active(app) {
            return Vec::new();
        }
        vec![TargetUse::new(
            OFFSCREEN,
            loaded_attachment(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        )]
    }

    unsafe fn record(&self, app: &App, cmd: vk::CommandBuffer, _: usize, _: usize) -> Result<()> {
        record_onion_skin_composite(app, cmd)
    }
}

const BLOOM_DOWNSAMPLE_NAMES: [&str; MAX_BLOOM_MIPS] = [
    "bloom_downsample_0",
    "bloom_downsample_1",
    "bloom_downsample_2",
    "bloom_downsample_3",
    "bloom_downsample_4",
    "bloom_downsample_5",
    "bloom_downsample_6",
    "bloom_downsample_7",
];

const BLOOM_UPSAMPLE_NAMES: [&str; MAX_BLOOM_MIPS] = [
    "bloom_upsample_0",
    "bloom_upsample_1",
    "bloom_upsample_2",
    "bloom_upsample_3",
    "bloom_upsample_4",
    "bloom_upsample_5",
    "bloom_upsample_6",
    "bloom_upsample_7",
];

static BLOOM_DOWNSAMPLE_NODES: [BloomDownsampleNode; MAX_BLOOM_MIPS] = [
    BloomDownsampleNode { mip_index: 0 },
    BloomDownsampleNode { mip_index: 1 },
    BloomDownsampleNode { mip_index: 2 },
    BloomDownsampleNode { mip_index: 3 },
    BloomDownsampleNode { mip_index: 4 },
    BloomDownsampleNode { mip_index: 5 },
    BloomDownsampleNode { mip_index: 6 },
    BloomDownsampleNode { mip_index: 7 },
];

static BLOOM_UPSAMPLE_NODES: [BloomUpsampleNode; MAX_BLOOM_MIPS] = [
    BloomUpsampleNode { pass_index: 0 },
    BloomUpsampleNode { pass_index: 1 },
    BloomUpsampleNode { pass_index: 2 },
    BloomUpsampleNode { pass_index: 3 },
    BloomUpsampleNode { pass_index: 4 },
    BloomUpsampleNode { pass_index: 5 },
    BloomUpsampleNode { pass_index: 6 },
    BloomUpsampleNode { pass_index: 7 },
];

pub fn register_core_passes(graph: &mut PassGraph) {
    graph.register(&CompositeHdrNode);
    graph.register(&OnionSkinNode);
    for node in &BLOOM_DOWNSAMPLE_NODES {
        graph.register(node);
    }
    for node in &BLOOM_UPSAMPLE_NODES {
        graph.register(node);
    }
    graph.register(&DofNode);
    graph.register(&AutoExposureNode);
    graph.register(&TonemapNode);
    graph.register(&OnionCompositeNode);
}
