use super::descriptor_allocator::{DescriptorAllocator, PoolSignature};
use super::swapchain::*;
use crate::vulkan::*;
use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::Deref;
use thiserror::Error;
use vulkanalia::Device as VulkanDevice;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationMode {
    Enabled,
    Disabled,
}

impl ValidationMode {
    pub fn is_enabled(self) -> bool {
        self == ValidationMode::Enabled
    }
}

pub const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

pub const HEADLESS_DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[
    vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION.name,
    vk::KHR_ACCELERATION_STRUCTURE_EXTENSION.name,
    vk::KHR_RAY_QUERY_EXTENSION.name,
    vk::KHR_DEFERRED_HOST_OPERATIONS_EXTENSION.name,
];

#[derive(Clone, Debug)]
pub struct Device(VulkanDevice);

impl Device {
    pub fn new(device: VulkanDevice) -> Self {
        Device(device)
    }
    pub fn null() -> Self {
        unsafe { Device(std::mem::zeroed()) }
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::null()
    }
}

impl Deref for Device {
    type Target = VulkanDevice;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Default)]
pub struct RRDevice {
    pub device: Device,
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub graphics_queue_family_index: u32,
    pub msaa_samples: vk::SampleCountFlags,
    pub min_uniform_buffer_offset_alignment: u64,
    pub timestamp_period: f32,
    descriptor_allocator: RefCell<DescriptorAllocator>,
}

impl RRDevice {
    pub unsafe fn default() -> RRDevice {
        Self {
            device: std::mem::zeroed(),
            physical_device: vk::PhysicalDevice::default(),
            graphics_queue: vk::Queue::default(),
            present_queue: vk::Queue::default(),
            graphics_queue_family_index: 0,
            msaa_samples: vk::SampleCountFlags::default(),
            min_uniform_buffer_offset_alignment: 256,
            timestamp_period: 1.0,
            descriptor_allocator: RefCell::default(),
        }
    }

    pub unsafe fn new(
        entry: &Entry,
        instance: &Instance,
        surface: &vk::SurfaceKHR,
        validation: ValidationMode,
        validation_layer: vk::ExtensionName,
        device_extensions: &[vk::ExtensionName],
        portability_macro_version: Version,
    ) -> Result<RRDevice> {
        let (physical_device, sample_count) =
            pick_physical_device(instance, surface, device_extensions)?;

        let graphics_index = GraphicsQueueIndex::find(instance, &physical_device)?;
        let present_index = PresentQueueIndex::find(instance, surface, &physical_device)?;

        let (device, graphics_queue, present_queue) = create_logical_device_with_present(
            entry,
            instance,
            graphics_index,
            present_index,
            validation,
            validation_layer,
            device_extensions,
            portability_macro_version,
            &physical_device,
        )?;

        let properties = instance.get_physical_device_properties(physical_device);
        let min_ubo_alignment = properties.limits.min_uniform_buffer_offset_alignment;
        let timestamp_period = properties.limits.timestamp_period;

        println!("created logical device");
        Ok(Self {
            device,
            physical_device,
            graphics_queue,
            present_queue,
            graphics_queue_family_index: graphics_index.0,
            msaa_samples: sample_count,
            min_uniform_buffer_offset_alignment: min_ubo_alignment,
            timestamp_period,
            descriptor_allocator: RefCell::default(),
        })
    }

    pub unsafe fn new_headless(
        entry: &Entry,
        instance: &Instance,
        device_extensions: &[vk::ExtensionName],
        validation: ValidationMode,
        validation_layer: vk::ExtensionName,
        portability_macro_version: Version,
    ) -> Result<RRDevice> {
        let (physical_device, sample_count) =
            pick_physical_device_headless(instance, device_extensions)?;

        let graphics_index = GraphicsQueueIndex::find(instance, &physical_device)?;

        let (device, graphics_queue) = create_logical_device_headless(
            entry,
            instance,
            graphics_index,
            validation,
            validation_layer,
            device_extensions,
            portability_macro_version,
            &physical_device,
        )?;

        let properties = instance.get_physical_device_properties(physical_device);
        let min_ubo_alignment = properties.limits.min_uniform_buffer_offset_alignment;
        let timestamp_period = properties.limits.timestamp_period;

        println!("created headless logical device");
        Ok(Self {
            device,
            physical_device,
            graphics_queue,
            present_queue: vk::Queue::null(),
            graphics_queue_family_index: graphics_index.0,
            msaa_samples: sample_count,
            min_uniform_buffer_offset_alignment: min_ubo_alignment,
            timestamp_period,
            descriptor_allocator: RefCell::default(),
        })
    }

    pub unsafe fn connect_surface(
        &mut self,
        instance: &Instance,
        surface: &vk::SurfaceKHR,
    ) -> Result<()> {
        let present_index = PresentQueueIndex::find(instance, surface, &self.physical_device)?;
        self.present_queue = self.device.get_device_queue(present_index.0, 0);
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct GraphicsQueueIndex(pub u32);

impl GraphicsQueueIndex {
    pub unsafe fn find(instance: &Instance, physical_device: &vk::PhysicalDevice) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(*physical_device);
        let index = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32)
            .ok_or_else(|| anyhow!(SuitabilityError("No graphics queue family")))?;
        Ok(Self(index))
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct PresentQueueIndex(pub u32);

impl PresentQueueIndex {
    pub unsafe fn find(
        instance: &Instance,
        surface: &vk::SurfaceKHR,
        physical_device: &vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(*physical_device);
        for (index, _) in properties.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(
                *physical_device,
                index as u32,
                *surface,
            )? {
                return Ok(Self(index as u32));
            }
        }
        Err(anyhow!(SuitabilityError("No present queue family")))
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub present: u32,
}

impl QueueFamilyIndices {
    pub unsafe fn get(
        instance: &Instance,
        surface: &vk::SurfaceKHR,
        physical_device: &vk::PhysicalDevice,
    ) -> Result<Self> {
        let graphics = GraphicsQueueIndex::find(instance, physical_device)?;
        let present = PresentQueueIndex::find(instance, surface, physical_device)?;
        Ok(Self {
            graphics: graphics.0,
            present: present.0,
        })
    }
}

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

unsafe fn create_device_common(
    entry: &Entry,
    instance: &Instance,
    queue_family_indices: &[u32],
    validation: ValidationMode,
    validation_layer: vk::ExtensionName,
    device_extensions: &[vk::ExtensionName],
    portability_macro_version: Version,
    physical_device: &vk::PhysicalDevice,
) -> Result<Device> {
    let queue_priorities = &[1.0];
    let queue_infos: Vec<_> = queue_family_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect();

    let layers = if validation.is_enabled() {
        vec![validation_layer.as_ptr()]
    } else {
        vec![]
    };

    let mut extensions: Vec<*const i8> = device_extensions.iter().map(|n| n.as_ptr()).collect();
    if cfg!(target_os = "macos") && entry.version()? >= portability_macro_version {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    let features = vk::PhysicalDeviceFeatures::builder()
        .sampler_anisotropy(true)
        .sample_rate_shading(true)
        .fill_mode_non_solid(true)
        .independent_blend(true)
        .geometry_shader(true);

    let mut accel_features =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder().acceleration_structure(true);
    let mut ray_query_features = vk::PhysicalDeviceRayQueryFeaturesKHR::builder().ray_query(true);
    let mut vulkan_12_features =
        vk::PhysicalDeviceVulkan12Features::builder().buffer_device_address(true);

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features)
        .push_next(&mut vulkan_12_features)
        .push_next(&mut accel_features)
        .push_next(&mut ray_query_features);

    let vulkan_device = instance.create_device(*physical_device, &info, None)?;
    Ok(Device::new(vulkan_device))
}

unsafe fn create_logical_device_headless(
    entry: &Entry,
    instance: &Instance,
    graphics_index: GraphicsQueueIndex,
    validation: ValidationMode,
    validation_layer: vk::ExtensionName,
    device_extensions: &[vk::ExtensionName],
    portability_macro_version: Version,
    physical_device: &vk::PhysicalDevice,
) -> Result<(Device, vk::Queue)> {
    let device = create_device_common(
        entry,
        instance,
        &[graphics_index.0],
        validation,
        validation_layer,
        device_extensions,
        portability_macro_version,
        physical_device,
    )?;

    let graphics_queue = device.get_device_queue(graphics_index.0, 0);
    Ok((device, graphics_queue))
}

unsafe fn create_logical_device_with_present(
    entry: &Entry,
    instance: &Instance,
    graphics_index: GraphicsQueueIndex,
    present_index: PresentQueueIndex,
    validation: ValidationMode,
    validation_layer: vk::ExtensionName,
    device_extensions: &[vk::ExtensionName],
    portability_macro_version: Version,
    physical_device: &vk::PhysicalDevice,
) -> Result<(Device, vk::Queue, vk::Queue)> {
    let mut unique_indices = HashSet::new();
    unique_indices.insert(graphics_index.0);
    unique_indices.insert(present_index.0);
    let queue_family_indices: Vec<u32> = unique_indices.into_iter().collect();

    let device = create_device_common(
        entry,
        instance,
        &queue_family_indices,
        validation,
        validation_layer,
        device_extensions,
        portability_macro_version,
        physical_device,
    )?;

    let graphics_queue = device.get_device_queue(graphics_index.0, 0);
    let present_queue = device.get_device_queue(present_index.0, 0);
    Ok((device, graphics_queue, present_queue))
}

const AMD_VENDOR_ID: u32 = 0x1002;

unsafe fn device_local_memory_bytes(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> u64 {
    let memory = instance.get_physical_device_memory_properties(physical_device);
    (0..memory.memory_heap_count as usize)
        .filter(|&i| {
            memory.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        })
        .map(|i| memory.memory_heaps[i].size)
        .sum()
}

// Enumeration order depends on the Vulkan loader / layer environment, so a
// first-suitable pick can land on a small display adapter. Rank every suitable
// device instead: THYLLORE_GPU name match > device-local VRAM > AMD (the render
// GPU on this workstation; the NVIDIA card stays free for compute).
fn device_rank(name_matches_env: bool, vram_bytes: u64, vendor_id: u32) -> (bool, u64, bool) {
    (name_matches_env, vram_bytes, vendor_id == AMD_VENDOR_ID)
}

fn preferred_gpu_name_lowercase() -> Option<String> {
    std::env::var("THYLLORE_GPU")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
}

unsafe fn pick_ranked_physical_device(
    instance: &Instance,
    surface: Option<&vk::SurfaceKHR>,
    device_extensions: &[vk::ExtensionName],
) -> Result<(vk::PhysicalDevice, vk::SampleCountFlags)> {
    let preferred_name = preferred_gpu_name_lowercase();
    let mut best: Option<(vk::PhysicalDevice, (bool, u64, bool))> = None;
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) =
            check_physical_device_capabilities(instance, &physical_device, device_extensions)
        {
            log!(
                "Skipping Physical Device (`{}`): {}",
                properties.device_name,
                error
            );
            continue;
        }

        if let Some(surface) = surface {
            if let Err(error) =
                check_physical_device_presentation(instance, surface, &physical_device)
            {
                log!(
                    "Skipping Physical Device (`{}`): {}",
                    properties.device_name,
                    error
                );
                continue;
            }
        }

        let name_matches_env = preferred_name.as_ref().is_some_and(|p| {
            properties
                .device_name
                .to_string()
                .to_lowercase()
                .contains(p)
        });
        let vram = device_local_memory_bytes(instance, physical_device);
        let rank = device_rank(name_matches_env, vram, properties.vendor_id);
        log!(
            "Suitable Physical Device (`{}`): {} MiB device-local{}",
            properties.device_name,
            vram / (1024 * 1024),
            if name_matches_env {
                ", matches THYLLORE_GPU"
            } else {
                ""
            }
        );
        if best.as_ref().is_none_or(|(_, best_rank)| rank > *best_rank) {
            best = Some((physical_device, rank));
        }
    }

    let (physical_device, _) = best.ok_or_else(|| {
        anyhow!(if surface.is_some() {
            "Failed to find suitable physical device"
        } else {
            "Failed to find suitable physical device (headless)"
        })
    })?;
    let properties = instance.get_physical_device_properties(physical_device);
    log!("Selected Physical Device (`{}`).", properties.device_name);
    let sample_count = get_max_msaa_samples(instance, physical_device);
    Ok((physical_device, sample_count))
}

unsafe fn pick_physical_device(
    instance: &Instance,
    surface: &vk::SurfaceKHR,
    device_extensions: &[vk::ExtensionName],
) -> Result<(vk::PhysicalDevice, vk::SampleCountFlags)> {
    pick_ranked_physical_device(instance, Some(surface), device_extensions)
}

unsafe fn pick_physical_device_headless(
    instance: &Instance,
    device_extensions: &[vk::ExtensionName],
) -> Result<(vk::PhysicalDevice, vk::SampleCountFlags)> {
    pick_ranked_physical_device(instance, None, device_extensions)
}

unsafe fn check_physical_device_capabilities(
    instance: &Instance,
    physical_device: &vk::PhysicalDevice,
    device_extensions: &[vk::ExtensionName],
) -> Result<()> {
    let properties = instance.get_physical_device_properties(*physical_device);
    log!("Checking device: {}", properties.device_name);
    log!("Device type: {:?}", properties.device_type);

    if properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
        log!("Device rejected: Only discrete GPUs are supported");
        return Err(anyhow!(SuitabilityError(
            "Only discrete GPUs are supported"
        )));
    }
    log!("Device type check passed");

    let features = instance.get_physical_device_features(*physical_device);
    if features.sampler_anisotropy != vk::TRUE {
        log!("Device rejected: No sampler anisotropy");
        return Err(anyhow!(SuitabilityError("No sampler anisotropy")));
    }
    log!("Sampler anisotropy check passed");

    if features.geometry_shader != vk::TRUE {
        log!("Device rejected: Missing geometry shader system");
        return Err(anyhow!(SuitabilityError(
            "Missing geometry shader supported"
        )));
    }
    log!("Geometry shader check passed");

    match GraphicsQueueIndex::find(instance, physical_device) {
        Ok(_) => log!("Queue family check passed"),
        Err(e) => {
            log!("Device rejected: Queue family check failed - {}", e);
            return Err(e);
        }
    }

    match check_physical_device_extensions(instance, physical_device, device_extensions) {
        Ok(_) => log!("Device extensions check passed"),
        Err(e) => {
            log!("Device rejected: Extensions check failed - {}", e);
            return Err(e);
        }
    }

    log!("All device capability checks passed!");
    Ok(())
}

unsafe fn check_physical_device_presentation(
    instance: &Instance,
    surface: &vk::SurfaceKHR,
    physical_device: &vk::PhysicalDevice,
) -> Result<()> {
    PresentQueueIndex::find(instance, surface, physical_device)?;
    log!("Present queue check passed");

    let support = SwapchainSupport::get(instance, surface, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        log!("Device rejected: Insufficient swapchain system");
        return Err(anyhow!(SuitabilityError("Insufficient swapchain system")));
    }
    log!("Swapchain system check passed");

    Ok(())
}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: &vk::PhysicalDevice,
    device_extensions: &[vk::ExtensionName],
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(*physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();

    log!("Required extensions: {}", device_extensions.len());
    for ext in device_extensions {
        let supported = extensions.contains(ext);
        log!(
            "  {} - {}",
            ext,
            if supported {
                "SUPPORTED"
            } else {
                "NOT SUPPORTED"
            }
        );
    }

    let missing_extensions: Vec<_> = device_extensions
        .iter()
        .filter(|e| !extensions.contains(e))
        .collect();

    if missing_extensions.is_empty() {
        Ok(())
    } else {
        log!("Missing {} extensions:", missing_extensions.len());
        for ext in &missing_extensions {
            log!("  - {}", ext);
        }
        Err(anyhow!(SuitabilityError("Device Extensions Not Supported")))
    }
}

unsafe fn get_max_msaa_samples(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::SampleCountFlags {
    let properties = instance.get_physical_device_properties(physical_device);
    let counts = properties.limits.framebuffer_color_sample_counts
        & properties.limits.framebuffer_depth_sample_counts;
    [
        vk::SampleCountFlags::_64,
        vk::SampleCountFlags::_32,
        vk::SampleCountFlags::_16,
        vk::SampleCountFlags::_8,
        vk::SampleCountFlags::_4,
        vk::SampleCountFlags::_2,
    ]
    .iter()
    .cloned()
    .find(|c| counts.contains(*c))
    .unwrap_or(vk::SampleCountFlags::_1)
}

pub unsafe fn create_headless_instance(entry: &Entry) -> Result<Instance> {
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Headless Test\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 2, 0));

    let validation_enabled = cfg!(debug_assertions);

    let mut extensions: Vec<*const i8> = Vec::new();
    if validation_enabled {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    let layers = if validation_enabled && available_layers.contains(&VALIDATION_LAYER) {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    let info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);

    let instance = entry.create_instance(&info, None)?;
    Ok(instance)
}

pub unsafe fn destroy_headless_instance(instance: &Instance) {
    instance.destroy_instance(None);
}

pub unsafe fn destroy_headless_device(device: &RRDevice, instance: &Instance) {
    device.device.device_wait_idle().ok();
    device.destroy_descriptor_pools();
    device.device.destroy_device(None);
    destroy_headless_instance(instance);
}

impl RRDevice {
    pub fn has_graphics_queue(&self) -> bool {
        self.graphics_queue != vk::Queue::null()
    }

    pub fn has_present_queue(&self) -> bool {
        self.present_queue != vk::Queue::null()
    }

    pub unsafe fn allocate_descriptor_sets(
        &self,
        layout: vk::DescriptorSetLayout,
        signature: &PoolSignature,
        count: usize,
    ) -> Result<Vec<vk::DescriptorSet>> {
        self.descriptor_allocator
            .borrow_mut()
            .allocate(&self.device, layout, signature, count)
    }

    pub unsafe fn destroy_descriptor_pools(&self) {
        self.descriptor_allocator.borrow_mut().destroy(&self.device);
    }

    pub unsafe fn wait_graphics_queue_idle(&self) -> Result<()> {
        self.device.queue_wait_idle(self.graphics_queue)?;
        Ok(())
    }

    pub unsafe fn query_physical_device_api_version(&self, instance: &Instance) -> u32 {
        let properties = instance.get_physical_device_properties(self.physical_device);
        properties.api_version
    }
}
