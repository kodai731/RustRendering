use std::collections::HashMap;

use crate::core::device::RRDevice;
use crate::resource::image::{create_image, create_image_view};
use crate::vulkan::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RenderTargetKey {
    SceneColorCopy,
    EffectHistory(u8),
    PostProcess(u8),
    TraceImage,
    CausticAccum,
}

#[derive(Clone, Copy, Debug)]
pub struct RenderTargetEntry {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
}

#[derive(Debug, Default)]
pub struct RenderTargetRegistry {
    entries: HashMap<RenderTargetKey, RenderTargetEntry>,
    width: u32,
    height: u32,
}

impl RenderTargetRegistry {
    pub unsafe fn ensure(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        key: RenderTargetKey,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        sampler_info: Option<vk::SamplerCreateInfo>,
    ) -> Result<&RenderTargetEntry> {
        if !self.has_valid_extent() {
            return Err(anyhow!(
                "render target {key:?} requested before set_extent_and_reset: extent is {}x{}",
                self.width,
                self.height
            ));
        }

        let is_reusable = self
            .entries
            .get(&key)
            .is_some_and(|entry| entry.format == format && entry.usage == usage);

        if !is_reusable {
            if let Some(outdated) = self.entries.remove(&key) {
                destroy_entry(&rrdevice.device, &outdated);
            }

            let entry = self.create_entry(instance, rrdevice, format, usage, sampler_info)?;
            self.entries.insert(key, entry);
        }

        self.entries
            .get(&key)
            .ok_or_else(|| anyhow!("render target {key:?} is missing after creation"))
    }

    unsafe fn create_entry(
        &self,
        instance: &Instance,
        rrdevice: &RRDevice,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        sampler_info: Option<vk::SamplerCreateInfo>,
    ) -> Result<RenderTargetEntry> {
        let (image, memory) = create_image(
            instance,
            rrdevice,
            self.width,
            self.height,
            1,
            vk::SampleCountFlags::_1,
            format,
            vk::ImageTiling::OPTIMAL,
            usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let view = match create_image_view(rrdevice, image, format, vk::ImageAspectFlags::COLOR, 1)
        {
            Ok(view) => view,
            Err(error) => {
                rrdevice.device.destroy_image(image, None);
                rrdevice.device.free_memory(memory, None);
                return Err(error);
            }
        };

        let sampler = match sampler_info {
            Some(info) => match rrdevice.device.create_sampler(&info, None) {
                Ok(sampler) => sampler,
                Err(error) => {
                    rrdevice.device.destroy_image_view(view, None);
                    rrdevice.device.destroy_image(image, None);
                    rrdevice.device.free_memory(memory, None);
                    return Err(error.into());
                }
            },
            None => vk::Sampler::null(),
        };

        Ok(RenderTargetEntry {
            image,
            memory,
            view,
            sampler,
            format,
            usage,
        })
    }

    pub fn get(&self, key: RenderTargetKey) -> Option<&RenderTargetEntry> {
        self.entries.get(&key)
    }

    pub fn extent(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub unsafe fn ensure_extent(&mut self, device: &vulkanalia::Device, width: u32, height: u32) {
        if (self.width, self.height) == (width, height) {
            return;
        }
        self.set_extent_and_reset(device, width, height);
    }

    pub unsafe fn set_extent_and_reset(
        &mut self,
        device: &vulkanalia::Device,
        width: u32,
        height: u32,
    ) {
        self.destroy_all(device);
        self.width = width;
        self.height = height;
    }

    pub unsafe fn destroy_all(&mut self, device: &vulkanalia::Device) {
        for (_, entry) in self.entries.drain() {
            destroy_entry(device, &entry);
        }
    }

    pub fn active_target_count(&self) -> usize {
        self.entries.len()
    }

    pub fn has_leaked_targets(&self) -> bool {
        !self.entries.is_empty()
    }

    pub fn clear_tracking(&mut self) {
        self.entries.clear();
    }

    fn has_valid_extent(&self) -> bool {
        self.width > 0 && self.height > 0
    }
}

impl Drop for RenderTargetRegistry {
    fn drop(&mut self) {
        if self.has_leaked_targets() {
            log_warn!(
                "RenderTargetRegistry dropped without calling destroy_all(): {} render targets leaked",
                self.active_target_count(),
            );
        }
    }
}

unsafe fn destroy_entry(device: &vulkanalia::Device, entry: &RenderTargetEntry) {
    if entry.sampler != vk::Sampler::null() {
        device.destroy_sampler(entry.sampler, None);
    }
    if entry.view != vk::ImageView::null() {
        device.destroy_image_view(entry.view, None);
    }
    if entry.image != vk::Image::null() {
        device.destroy_image(entry.image, None);
    }
    if entry.memory != vk::DeviceMemory::null() {
        device.free_memory(entry.memory, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert_dummy(registry: &mut RenderTargetRegistry, key: RenderTargetKey) {
        registry.entries.insert(
            key,
            RenderTargetEntry {
                image: vk::Image::null(),
                memory: vk::DeviceMemory::null(),
                view: vk::ImageView::null(),
                sampler: vk::Sampler::null(),
                format: vk::Format::R16G16B16A16_SFLOAT,
                usage: vk::ImageUsageFlags::STORAGE,
            },
        );
    }

    #[test]
    fn test_default_registry_is_empty_and_has_no_extent() {
        let registry = RenderTargetRegistry::default();

        assert_eq!(registry.extent(), (0, 0));
        assert!(!registry.has_valid_extent());
        assert!(!registry.has_leaked_targets());
        assert!(registry.get(RenderTargetKey::TraceImage).is_none());
    }

    #[test]
    fn test_effect_history_indices_are_distinct_keys() {
        let mut registry = RenderTargetRegistry::default();
        insert_dummy(&mut registry, RenderTargetKey::EffectHistory(0));
        insert_dummy(&mut registry, RenderTargetKey::EffectHistory(1));

        assert_eq!(registry.active_target_count(), 2);
        assert!(registry.get(RenderTargetKey::EffectHistory(0)).is_some());
        assert!(registry.get(RenderTargetKey::EffectHistory(1)).is_some());
        assert!(registry.get(RenderTargetKey::EffectHistory(2)).is_none());

        registry.clear_tracking();
    }

    #[test]
    fn test_inserting_same_key_twice_keeps_single_entry() {
        let mut registry = RenderTargetRegistry::default();
        insert_dummy(&mut registry, RenderTargetKey::CausticAccum);
        insert_dummy(&mut registry, RenderTargetKey::CausticAccum);

        assert_eq!(registry.active_target_count(), 1);

        registry.clear_tracking();
    }

    #[test]
    fn test_clear_tracking_prevents_leak_report() {
        let mut registry = RenderTargetRegistry::default();
        insert_dummy(&mut registry, RenderTargetKey::SceneColorCopy);
        insert_dummy(&mut registry, RenderTargetKey::TraceImage);

        assert!(registry.has_leaked_targets());

        registry.clear_tracking();

        assert!(!registry.has_leaked_targets());
        assert_eq!(registry.active_target_count(), 0);
    }
}
