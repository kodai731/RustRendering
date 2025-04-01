pub use anyhow::{anyhow, Result};
pub use core::result::Result::Ok;
pub use log::*;
pub use vulkanalia::loader::{LibloadingLoader, LIBRARY};
pub use vulkanalia::prelude::v1_2::*;
pub use vulkanalia::vk::ExtDebugUtilsExtension;
pub use vulkanalia::vk::KhrSurfaceExtension;
pub use vulkanalia::vk::KhrSwapchainExtension;
pub use vulkanalia::window as vk_window;
pub use vulkanalia::Version;

pub unsafe fn get_memory_type_index(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Result<u32> {
    let memory = instance.get_physical_device_memory_properties(physical_device);
    (0..memory.memory_type_count)
        .find(|i| {
            let suitable = (requirements.memory_type_bits & (1 << i) as u32) != 0;
            let memory_type = memory.memory_types[*i as usize];
            suitable & memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
}
