use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::pass_recording::*;
use crate::app::App;
use crate::hooks::pass::{PassGraph, PassStage, RenderPassNode};

type CoreRecordFn = unsafe fn(&App, vk::CommandBuffer, usize, usize) -> Result<()>;

pub struct CorePassNode {
    name: &'static str,
    stage: PassStage,
    record: CoreRecordFn,
}

impl RenderPassNode for CorePassNode {
    fn name(&self) -> &'static str {
        self.name
    }

    fn stage(&self) -> PassStage {
        self.stage
    }

    unsafe fn record(
        &self,
        app: &App,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        frame_slot: usize,
    ) -> Result<()> {
        (self.record)(app, command_buffer, image_index, frame_slot)
    }
}

unsafe fn composite_hdr(app: &App, cmd: vk::CommandBuffer, _: usize, _: usize) -> Result<()> {
    record_composite_to_hdr(app, cmd)
}

unsafe fn onion_skin(
    app: &App,
    cmd: vk::CommandBuffer,
    image_index: usize,
    _: usize,
) -> Result<()> {
    record_onion_skin_pass(app, cmd, image_index)
}

unsafe fn bloom(app: &App, cmd: vk::CommandBuffer, _: usize, frame_slot: usize) -> Result<()> {
    record_bloom(app, cmd, frame_slot)
}

unsafe fn dof(app: &App, cmd: vk::CommandBuffer, _: usize, _: usize) -> Result<()> {
    record_dof(app, cmd)
}

unsafe fn auto_exposure(
    app: &App,
    cmd: vk::CommandBuffer,
    _: usize,
    frame_slot: usize,
) -> Result<()> {
    record_auto_exposure(app, cmd, frame_slot)
}

unsafe fn tonemap(app: &App, cmd: vk::CommandBuffer, image_index: usize, _: usize) -> Result<()> {
    record_tonemap_to_offscreen(app, cmd, image_index)
}

unsafe fn onion_composite(app: &App, cmd: vk::CommandBuffer, _: usize, _: usize) -> Result<()> {
    record_onion_skin_composite(app, cmd)
}

const CORE_PASSES: &[CorePassNode] = &[
    CorePassNode {
        name: "composite_hdr",
        stage: PassStage::Lighting,
        record: composite_hdr,
    },
    CorePassNode {
        name: "onion_skin",
        stage: PassStage::Lighting,
        record: onion_skin,
    },
    CorePassNode {
        name: "bloom",
        stage: PassStage::PostProcess,
        record: bloom,
    },
    CorePassNode {
        name: "dof",
        stage: PassStage::PostProcess,
        record: dof,
    },
    CorePassNode {
        name: "auto_exposure",
        stage: PassStage::PostProcess,
        record: auto_exposure,
    },
    CorePassNode {
        name: "tonemap",
        stage: PassStage::Final,
        record: tonemap,
    },
    CorePassNode {
        name: "onion_composite",
        stage: PassStage::Final,
        record: onion_composite,
    },
];

pub fn register_core_passes(graph: &mut PassGraph) {
    for node in CORE_PASSES {
        graph.register(node);
    }
}
