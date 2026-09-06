use std::collections::HashMap;

use crate::resource::{RenderTargetKey, TransientHandle};
use crate::vulkan::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CoreTarget {
    HdrColor,
    OnionSkinGhost,
    GBufferPosition,
    GBufferNormal,
    GBufferAlbedo,
    GBufferObjectId,
    GBufferShadowMask,
    Offscreen,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TargetRef {
    Storage(RenderTargetKey),
    Transient(TransientHandle),
    Core(CoreTarget),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShaderStage {
    Fragment,
    Compute,
    RayTracing,
}

impl ShaderStage {
    fn pipeline_stage(self) -> vk::PipelineStageFlags {
        match self {
            ShaderStage::Fragment => vk::PipelineStageFlags::FRAGMENT_SHADER,
            ShaderStage::Compute => vk::PipelineStageFlags::COMPUTE_SHADER,
            ShaderStage::RayTracing => vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TargetAccess {
    Attachment {
        initial_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    },
    Sampled(ShaderStage),
    StorageRead(ShaderStage),
    StorageReadWrite(ShaderStage),
    TransferSrc,
    TransferDst,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TargetUse {
    pub target: TargetRef,
    pub access: TargetAccess,
}

impl TargetUse {
    pub const fn new(target: TargetRef, access: TargetAccess) -> Self {
        Self { target, access }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ImageState {
    layout: vk::ImageLayout,
    stage: vk::PipelineStageFlags,
    access: vk::AccessFlags,
    writes: bool,
}

const UNKNOWN_STATE: ImageState = ImageState {
    layout: vk::ImageLayout::UNDEFINED,
    stage: vk::PipelineStageFlags::TOP_OF_PIPE,
    access: vk::AccessFlags::empty(),
    writes: false,
};

impl TargetAccess {
    fn entry_state(self) -> ImageState {
        match self {
            TargetAccess::Attachment { initial_layout, .. } => ImageState {
                layout: initial_layout,
                stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                writes: true,
            },
            TargetAccess::Sampled(stage) => ImageState {
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                stage: stage.pipeline_stage(),
                access: vk::AccessFlags::SHADER_READ,
                writes: false,
            },
            TargetAccess::StorageRead(stage) => ImageState {
                layout: vk::ImageLayout::GENERAL,
                stage: stage.pipeline_stage(),
                access: vk::AccessFlags::SHADER_READ,
                writes: false,
            },
            TargetAccess::StorageReadWrite(stage) => ImageState {
                layout: vk::ImageLayout::GENERAL,
                stage: stage.pipeline_stage(),
                access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                writes: true,
            },
            TargetAccess::TransferSrc => ImageState {
                layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                stage: vk::PipelineStageFlags::TRANSFER,
                access: vk::AccessFlags::TRANSFER_READ,
                writes: false,
            },
            TargetAccess::TransferDst => ImageState {
                layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                stage: vk::PipelineStageFlags::TRANSFER,
                access: vk::AccessFlags::TRANSFER_WRITE,
                writes: true,
            },
        }
    }

    fn exit_state(self) -> ImageState {
        let mut state = self.entry_state();
        if let TargetAccess::Attachment { final_layout, .. } = self {
            state.layout = final_layout;
        }
        state
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PendingBarrier {
    pub src_stage: vk::PipelineStageFlags,
    pub dst_stage: vk::PipelineStageFlags,
    pub image: vk::Image,
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
    pub src_access: vk::AccessFlags,
    pub dst_access: vk::AccessFlags,
}

impl PendingBarrier {
    pub fn image_memory_barrier(&self) -> vk::ImageMemoryBarrier {
        vk::ImageMemoryBarrier::builder()
            .image(self.image)
            .old_layout(self.old_layout)
            .new_layout(self.new_layout)
            .src_access_mask(self.src_access)
            .dst_access_mask(self.dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build()
    }
}

/// Tracks the layout and last access of every color image the pass graph touches
/// and derives the barrier a pass needs before it runs.
#[derive(Debug, Default)]
pub struct ImageStateTracker {
    states: HashMap<vk::Image, ImageState>,
}

impl ImageStateTracker {
    pub fn forget(&mut self, image: vk::Image) {
        self.states.remove(&image);
    }

    /// Records an image that was transitioned to SHADER_READ_ONLY_OPTIMAL outside the graph
    /// (history images cleared at creation), so its first read needs no barrier.
    pub fn mark_shader_read_only(&mut self, image: vk::Image) {
        self.states.insert(
            image,
            ImageState {
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                access: vk::AccessFlags::SHADER_READ,
                writes: false,
            },
        );
    }

    pub fn clear(&mut self) {
        self.states.clear();
    }

    pub fn transition(&mut self, image: vk::Image, access: TargetAccess) -> Option<PendingBarrier> {
        let current = self.states.get(&image).copied().unwrap_or(UNKNOWN_STATE);
        let entry = access.entry_state();
        self.states.insert(image, access.exit_state());

        let render_pass_discards_contents = matches!(
            access,
            TargetAccess::Attachment {
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..
            }
        );
        if render_pass_discards_contents {
            return None;
        }

        let hazard = current.layout != entry.layout || current.writes || entry.writes;
        if !hazard {
            return None;
        }

        Some(PendingBarrier {
            src_stage: current.stage,
            dst_stage: entry.stage,
            image,
            old_layout: current.layout,
            new_layout: entry.layout,
            src_access: current.access,
            dst_access: entry.access,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(id: u64) -> vk::Image {
        vk::Image::from_raw(id)
    }

    #[test]
    fn first_use_transitions_from_undefined() {
        let mut tracker = ImageStateTracker::default();
        let barrier = tracker
            .transition(image(1), TargetAccess::TransferDst)
            .expect("barrier");
        assert_eq!(barrier.old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(barrier.new_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        assert_eq!(barrier.src_stage, vk::PipelineStageFlags::TOP_OF_PIPE);
    }

    #[test]
    fn read_after_read_in_same_layout_needs_no_barrier() {
        let mut tracker = ImageStateTracker::default();
        tracker.transition(image(1), TargetAccess::Sampled(ShaderStage::Fragment));
        assert!(tracker
            .transition(image(1), TargetAccess::Sampled(ShaderStage::Compute))
            .is_none());
    }

    #[test]
    fn write_after_read_in_same_layout_needs_barrier() {
        let mut tracker = ImageStateTracker::default();
        tracker.transition(image(1), TargetAccess::StorageRead(ShaderStage::Fragment));
        let barrier = tracker
            .transition(
                image(1),
                TargetAccess::StorageReadWrite(ShaderStage::Compute),
            )
            .expect("barrier");
        assert_eq!(barrier.old_layout, vk::ImageLayout::GENERAL);
        assert_eq!(barrier.new_layout, vk::ImageLayout::GENERAL);
        assert_eq!(barrier.src_stage, vk::PipelineStageFlags::FRAGMENT_SHADER);
        assert_eq!(barrier.dst_stage, vk::PipelineStageFlags::COMPUTE_SHADER);
    }

    #[test]
    fn attachment_with_undefined_initial_layout_skips_barrier_and_lands_on_final() {
        let mut tracker = ImageStateTracker::default();
        tracker.transition(image(1), TargetAccess::TransferDst);
        let attachment = TargetAccess::Attachment {
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        assert!(tracker.transition(image(1), attachment).is_none());
        let barrier = tracker
            .transition(image(1), TargetAccess::TransferSrc)
            .expect("barrier");
        assert_eq!(
            barrier.old_layout,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
        assert_eq!(
            barrier.src_stage,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        );
    }

    #[test]
    fn attachment_with_loaded_contents_transitions_to_initial_layout() {
        let mut tracker = ImageStateTracker::default();
        tracker.transition(
            image(1),
            TargetAccess::StorageReadWrite(ShaderStage::Compute),
        );
        let attachment = TargetAccess::Attachment {
            initial_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let barrier = tracker.transition(image(1), attachment).expect("barrier");
        assert_eq!(barrier.old_layout, vk::ImageLayout::GENERAL);
        assert_eq!(
            barrier.new_layout,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
    }

    #[test]
    fn marked_shader_read_only_image_is_read_without_barrier() {
        let mut tracker = ImageStateTracker::default();
        tracker.mark_shader_read_only(image(1));
        assert!(tracker
            .transition(image(1), TargetAccess::Sampled(ShaderStage::Fragment))
            .is_none());
    }

    #[test]
    fn forget_resets_to_undefined() {
        let mut tracker = ImageStateTracker::default();
        tracker.transition(image(1), TargetAccess::Sampled(ShaderStage::Fragment));
        tracker.forget(image(1));
        let barrier = tracker
            .transition(image(1), TargetAccess::Sampled(ShaderStage::Fragment))
            .expect("barrier");
        assert_eq!(barrier.old_layout, vk::ImageLayout::UNDEFINED);
    }
}
