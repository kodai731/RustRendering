use std::collections::HashSet;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::App;
use crate::hooks::pass::{CoreTarget, RenderPassNode, TargetRef, TargetUse};
use thyllore_vulkan_core::renderer::PendingBarrier;

impl App {
    fn resolve_target_image(&self, target: TargetRef) -> Result<vk::Image> {
        match target {
            TargetRef::Core(core) => self.resolve_core_target_image(core),
            TargetRef::Storage(key) => self
                .data
                .viewport
                .storage
                .get(key)
                .map(|entry| entry.image)
                .ok_or_else(|| anyhow::anyhow!("storage target {key:?} is not allocated")),
            TargetRef::Transient(handle) => Ok(self.data.viewport.transient.get(handle)?.image),
        }
    }

    fn resolve_core_target_image(&self, core: CoreTarget) -> Result<vk::Image> {
        let image = match core {
            CoreTarget::HdrColor => self
                .data
                .viewport
                .hdr_buffer
                .as_ref()
                .map(|hdr| hdr.color_image),
            CoreTarget::OnionSkinGhost => self
                .data
                .raytracing
                .onion_skin_pass
                .as_ref()
                .map(|onion| onion.ghost_image),
            CoreTarget::GBufferPosition => self.gbuffer_image(|g| g.position_image),
            CoreTarget::GBufferNormal => self.gbuffer_image(|g| g.normal_image),
            CoreTarget::GBufferAlbedo => self.gbuffer_image(|g| g.albedo_image),
            CoreTarget::GBufferObjectId => self.gbuffer_image(|g| g.object_id_image),
            CoreTarget::GBufferShadowMask => self.gbuffer_image(|g| g.shadow_mask_image),
            CoreTarget::Offscreen => self
                .data
                .viewport
                .offscreen
                .as_ref()
                .map(|offscreen| offscreen.resolve_color_image),
        };
        image.ok_or_else(|| anyhow::anyhow!("core target {core:?} is not allocated"))
    }

    fn gbuffer_image(
        &self,
        select: fn(&thyllore_vulkan_core::resource::RRGBuffer) -> vk::Image,
    ) -> Option<vk::Image> {
        self.data.raytracing.gbuffer.as_ref().map(select)
    }

    pub(super) fn collect_pass_barriers(
        &mut self,
        node: &dyn RenderPassNode,
        transients_seen_this_frame: &mut HashSet<vk::Image>,
    ) -> Result<Vec<PendingBarrier>> {
        let uses: Vec<TargetUse> = node
            .reads(self)
            .into_iter()
            .chain(node.writes(self))
            .collect();

        let mut barriers = Vec::new();
        for target_use in uses {
            let image = self.resolve_target_image(target_use.target)?;
            let acquired_this_frame = matches!(target_use.target, TargetRef::Transient(_))
                && transients_seen_this_frame.insert(image);
            if acquired_this_frame {
                self.data.pass_image_states.forget(image);
            }
            if let Some(barrier) = self
                .data
                .pass_image_states
                .transition(image, target_use.access)
            {
                barriers.push(barrier);
            }
        }
        Ok(barriers)
    }

    pub(super) unsafe fn record_pass_barriers(
        &self,
        command_buffer: vk::CommandBuffer,
        barriers: &[PendingBarrier],
    ) {
        if barriers.is_empty() {
            return;
        }

        let src_stage = barriers
            .iter()
            .fold(vk::PipelineStageFlags::empty(), |acc, b| acc | b.src_stage);
        let dst_stage = barriers
            .iter()
            .fold(vk::PipelineStageFlags::empty(), |acc, b| acc | b.dst_stage);
        let image_barriers: Vec<vk::ImageMemoryBarrier> = barriers
            .iter()
            .map(PendingBarrier::image_memory_barrier)
            .collect();

        self.rrdevice.device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &image_barriers,
        );
    }
}
