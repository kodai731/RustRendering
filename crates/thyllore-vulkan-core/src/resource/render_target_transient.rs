use std::collections::HashMap;

use crate::core::device::RRDevice;
use crate::resource::image::{create_image, create_image_view};
use crate::vulkan::*;

pub const TRANSIENT_EVICT_AFTER_FRAMES: u64 = 4;
const MAX_FRAMEBUFFER_ATTACHMENTS: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TransientDesc {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TransientHandle {
    slot: u32,
    frame: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct TransientImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub desc: TransientDesc,
    pub generation: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PoolState {
    Free,
    InUse { bucket: usize },
}

#[derive(Clone, Copy, Debug)]
struct PooledImage {
    desc: TransientDesc,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    generation: u64,
    last_used_frame: u64,
    last_bucket: usize,
    state: PoolState,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct FramebufferKey {
    render_pass: vk::RenderPass,
    views: [vk::ImageView; MAX_FRAMEBUFFER_ATTACHMENTS],
    view_count: u8,
}

#[derive(Debug)]
pub struct RenderTargetTransient {
    pool: Vec<Option<PooledImage>>,
    buckets: Vec<Vec<u32>>,
    current_bucket: usize,
    current_frame: u64,
    next_generation: u64,
    framebuffers: HashMap<FramebufferKey, vk::Framebuffer>,
}

impl Default for RenderTargetTransient {
    fn default() -> Self {
        Self::new(2)
    }
}

impl RenderTargetTransient {
    pub fn new(frames_in_flight: usize) -> Self {
        Self {
            pool: Vec::new(),
            buckets: vec![Vec::new(); frames_in_flight.max(1)],
            current_bucket: 0,
            current_frame: 0,
            next_generation: 1,
            framebuffers: HashMap::new(),
        }
    }

    pub unsafe fn begin_frame(&mut self, device: &Device, frame_index: usize) -> Result<()> {
        let evicted = self.advance_frame(frame_index)?;

        for pooled in evicted {
            self.destroy_framebuffers_using(device, pooled.view);
            destroy_pooled(device, &pooled);
        }

        Ok(())
    }

    fn advance_frame(&mut self, frame_index: usize) -> Result<Vec<PooledImage>> {
        if frame_index >= self.buckets.len() {
            return Err(anyhow!(
                "transient frame index {frame_index} exceeds {} frames in flight",
                self.buckets.len()
            ));
        }

        for slot in self.buckets[frame_index].drain(..) {
            if let Some(Some(pooled)) = self.pool.get_mut(slot as usize) {
                pooled.state = PoolState::Free;
            }
        }

        self.current_bucket = frame_index;
        self.current_frame += 1;

        Ok(self.take_evictable())
    }

    fn take_evictable(&mut self) -> Vec<PooledImage> {
        let current_frame = self.current_frame;
        let mut evicted = Vec::new();

        for entry in self.pool.iter_mut() {
            let is_stale = entry.is_some_and(|pooled| {
                pooled.state == PoolState::Free
                    && current_frame - pooled.last_used_frame > TRANSIENT_EVICT_AFTER_FRAMES
            });
            if is_stale {
                if let Some(pooled) = entry.take() {
                    evicted.push(pooled);
                }
            }
        }

        evicted
    }

    pub unsafe fn acquire(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        desc: TransientDesc,
    ) -> Result<TransientHandle> {
        if let Some(handle) = self.acquire_pooled(desc) {
            return Ok(handle);
        }

        let (image, memory) = create_image(
            instance,
            rrdevice,
            desc.width,
            desc.height,
            1,
            vk::SampleCountFlags::_1,
            desc.format,
            vk::ImageTiling::OPTIMAL,
            desc.usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let view =
            match create_image_view(rrdevice, image, desc.format, vk::ImageAspectFlags::COLOR, 1) {
                Ok(view) => view,
                Err(error) => {
                    rrdevice.device.destroy_image(image, None);
                    rrdevice.device.free_memory(memory, None);
                    return Err(error);
                }
            };

        Ok(self.register(desc, image, memory, view))
    }

    fn acquire_pooled(&mut self, desc: TransientDesc) -> Option<TransientHandle> {
        let is_free_match =
            |pooled: &PooledImage| pooled.state == PoolState::Free && pooled.desc == desc;
        let current_bucket = self.current_bucket;

        let same_bucket = self.pool.iter().position(|entry| {
            entry.is_some_and(|pooled| {
                is_free_match(&pooled) && pooled.last_bucket == current_bucket
            })
        });
        let slot = same_bucket.or_else(|| {
            self.pool
                .iter()
                .position(|entry| entry.is_some_and(|pooled| is_free_match(&pooled)))
        })?;

        self.mark_in_use(slot);
        Some(self.handle_for(slot))
    }

    fn register(
        &mut self,
        desc: TransientDesc,
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
    ) -> TransientHandle {
        let pooled = PooledImage {
            desc,
            image,
            memory,
            view,
            generation: self.next_generation,
            last_used_frame: self.current_frame,
            last_bucket: self.current_bucket,
            state: PoolState::Free,
        };
        self.next_generation += 1;

        let slot = match self.pool.iter().position(Option::is_none) {
            Some(slot) => {
                self.pool[slot] = Some(pooled);
                slot
            }
            None => {
                self.pool.push(Some(pooled));
                self.pool.len() - 1
            }
        };

        self.mark_in_use(slot);
        self.handle_for(slot)
    }

    fn mark_in_use(&mut self, slot: usize) {
        if let Some(Some(pooled)) = self.pool.get_mut(slot) {
            pooled.state = PoolState::InUse {
                bucket: self.current_bucket,
            };
            pooled.last_used_frame = self.current_frame;
            pooled.last_bucket = self.current_bucket;
        }
        self.buckets[self.current_bucket].push(slot as u32);
    }

    fn handle_for(&self, slot: usize) -> TransientHandle {
        TransientHandle {
            slot: slot as u32,
            frame: self.current_frame,
        }
    }

    pub fn get(&self, handle: TransientHandle) -> Result<TransientImage> {
        if handle.frame != self.current_frame {
            return Err(anyhow!(
                "transient handle from frame {} used in frame {}",
                handle.frame,
                self.current_frame
            ));
        }

        let pooled = self
            .pool
            .get(handle.slot as usize)
            .copied()
            .flatten()
            .ok_or_else(|| anyhow!("transient slot {} is not allocated", handle.slot))?;

        Ok(TransientImage {
            image: pooled.image,
            view: pooled.view,
            desc: pooled.desc,
            generation: pooled.generation,
        })
    }

    pub unsafe fn framebuffer(
        &mut self,
        device: &Device,
        render_pass: vk::RenderPass,
        views: &[vk::ImageView],
        width: u32,
        height: u32,
    ) -> Result<vk::Framebuffer> {
        let key = FramebufferKey::new(render_pass, views)?;

        if let Some(framebuffer) = self.framebuffers.get(&key) {
            return Ok(*framebuffer);
        }

        let info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(views)
            .width(width)
            .height(height)
            .layers(1);
        let framebuffer = device.create_framebuffer(&info, None)?;
        self.framebuffers.insert(key, framebuffer);

        Ok(framebuffer)
    }

    unsafe fn destroy_framebuffers_using(&mut self, device: &Device, view: vk::ImageView) {
        let stale: Vec<FramebufferKey> = self
            .framebuffers
            .keys()
            .filter(|key| key.contains(view))
            .copied()
            .collect();

        for key in stale {
            if let Some(framebuffer) = self.framebuffers.remove(&key) {
                device.destroy_framebuffer(framebuffer, None);
            }
        }
    }

    pub unsafe fn destroy_all(&mut self, device: &Device) {
        for (_, framebuffer) in self.framebuffers.drain() {
            device.destroy_framebuffer(framebuffer, None);
        }

        for pooled in self.pool.drain(..).flatten() {
            destroy_pooled(device, &pooled);
        }

        for bucket in self.buckets.iter_mut() {
            bucket.clear();
        }
    }

    pub fn active_count(&self) -> usize {
        self.pool
            .iter()
            .flatten()
            .filter(|pooled| matches!(pooled.state, PoolState::InUse { .. }))
            .count()
    }

    pub fn pooled_count(&self) -> usize {
        self.pool.iter().flatten().count()
    }

    pub fn framebuffer_count(&self) -> usize {
        self.framebuffers.len()
    }

    pub fn clear_tracking(&mut self) {
        self.pool.clear();
        self.framebuffers.clear();
        for bucket in self.buckets.iter_mut() {
            bucket.clear();
        }
    }
}

impl Drop for RenderTargetTransient {
    fn drop(&mut self) {
        let pooled = self.pooled_count();
        if pooled > 0 || !self.framebuffers.is_empty() {
            log_warn!(
                "RenderTargetTransient dropped without calling destroy_all(): {} images, {} framebuffers leaked",
                pooled,
                self.framebuffers.len(),
            );
        }
    }
}

impl FramebufferKey {
    fn new(render_pass: vk::RenderPass, views: &[vk::ImageView]) -> Result<Self> {
        if views.is_empty() || views.len() > MAX_FRAMEBUFFER_ATTACHMENTS {
            return Err(anyhow!(
                "framebuffer attachment count {} is outside 1..={MAX_FRAMEBUFFER_ATTACHMENTS}",
                views.len()
            ));
        }

        let mut padded = [vk::ImageView::null(); MAX_FRAMEBUFFER_ATTACHMENTS];
        padded[..views.len()].copy_from_slice(views);

        Ok(Self {
            render_pass,
            views: padded,
            view_count: views.len() as u8,
        })
    }

    fn contains(&self, view: vk::ImageView) -> bool {
        self.views[..self.view_count as usize].contains(&view)
    }
}

unsafe fn destroy_pooled(device: &Device, pooled: &PooledImage) {
    if pooled.view != vk::ImageView::null() {
        device.destroy_image_view(pooled.view, None);
    }
    if pooled.image != vk::Image::null() {
        device.destroy_image(pooled.image, None);
    }
    if pooled.memory != vk::DeviceMemory::null() {
        device.free_memory(pooled.memory, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hdr_desc(width: u32, height: u32) -> TransientDesc {
        TransientDesc {
            width,
            height,
            format: vk::Format::R16G16B16A16_SFLOAT,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        }
    }

    fn acquire_null(transient: &mut RenderTargetTransient, desc: TransientDesc) -> TransientHandle {
        transient.acquire_pooled(desc).unwrap_or_else(|| {
            transient.register(
                desc,
                vk::Image::null(),
                vk::DeviceMemory::null(),
                vk::ImageView::null(),
            )
        })
    }

    #[test]
    fn test_same_desc_twice_in_one_frame_yields_distinct_slots() {
        let mut transient = RenderTargetTransient::new(2);
        transient.advance_frame(0).unwrap();

        let first = acquire_null(&mut transient, hdr_desc(64, 64));
        let second = acquire_null(&mut transient, hdr_desc(64, 64));

        assert_ne!(first.slot, second.slot);
        assert_eq!(transient.active_count(), 2);
        transient.clear_tracking();
    }

    #[test]
    fn test_handle_from_previous_frame_is_rejected() {
        let mut transient = RenderTargetTransient::new(2);
        transient.advance_frame(0).unwrap();
        let handle = acquire_null(&mut transient, hdr_desc(64, 64));
        assert!(transient.get(handle).is_ok());

        transient.advance_frame(1).unwrap();

        assert!(transient.get(handle).is_err());
        transient.clear_tracking();
    }

    #[test]
    fn test_bucket_is_reused_after_frames_in_flight_wrap() {
        let mut transient = RenderTargetTransient::new(2);
        transient.advance_frame(0).unwrap();
        let first = acquire_null(&mut transient, hdr_desc(64, 64));

        transient.advance_frame(1).unwrap();
        let second = acquire_null(&mut transient, hdr_desc(64, 64));
        assert_ne!(first.slot, second.slot);

        transient.advance_frame(0).unwrap();
        let third = acquire_null(&mut transient, hdr_desc(64, 64));

        assert_eq!(third.slot, first.slot);
        assert_eq!(transient.pooled_count(), 2);
        transient.clear_tracking();
    }

    #[test]
    fn test_same_bucket_keeps_same_slot_in_steady_state() {
        let mut transient = RenderTargetTransient::new(2);
        let mut slots_by_bucket = [Vec::new(), Vec::new()];

        for frame in 0..8 {
            let bucket = frame % 2;
            transient.advance_frame(bucket).unwrap();
            let handle = acquire_null(&mut transient, hdr_desc(64, 64));
            slots_by_bucket[bucket].push(handle.slot);
        }

        for slots in &slots_by_bucket {
            assert!(slots.windows(2).all(|pair| pair[0] == pair[1]));
        }
        assert_ne!(slots_by_bucket[0][0], slots_by_bucket[1][0]);
        assert_eq!(transient.pooled_count(), 2);
        transient.clear_tracking();
    }

    #[test]
    fn test_stale_free_image_is_evicted() {
        let mut transient = RenderTargetTransient::new(2);
        transient.advance_frame(0).unwrap();
        acquire_null(&mut transient, hdr_desc(64, 64));

        let mut evicted_total = 0;
        for frame in 1..=(TRANSIENT_EVICT_AFTER_FRAMES as usize + 2) {
            evicted_total += transient.advance_frame(frame % 2).unwrap().len();
        }

        assert_eq!(evicted_total, 1);
        assert_eq!(transient.pooled_count(), 0);
    }

    #[test]
    fn test_mismatched_extent_is_not_reused() {
        let mut transient = RenderTargetTransient::new(2);
        transient.advance_frame(0).unwrap();
        let small = acquire_null(&mut transient, hdr_desc(64, 64));

        transient.advance_frame(1).unwrap();
        transient.advance_frame(0).unwrap();
        let large = acquire_null(&mut transient, hdr_desc(128, 128));

        assert_ne!(small.slot, large.slot);
        assert_eq!(transient.pooled_count(), 2);
        transient.clear_tracking();
    }

    #[test]
    fn test_frame_index_out_of_range_is_error() {
        let mut transient = RenderTargetTransient::new(2);
        assert!(transient.advance_frame(2).is_err());
    }

    #[test]
    fn test_framebuffer_key_rejects_empty_and_oversized_attachments() {
        let render_pass = vk::RenderPass::null();
        assert!(FramebufferKey::new(render_pass, &[]).is_err());
        assert!(FramebufferKey::new(render_pass, &[vk::ImageView::null(); 5]).is_err());
        assert!(FramebufferKey::new(render_pass, &[vk::ImageView::null(); 4]).is_ok());
    }
}
