use super::device::*;
use super::image::*;
use super::render::*;
use super::vulkan::*;
use super::window::*;
use crate::vulkanr::command::RRCommandPool;
use crate::AppData;

#[derive(Clone, Debug)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

#[derive(Clone, Debug, Default)]
pub struct RRSwapchain {
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_image_views: Vec<vk::ImageView>,
}

impl RRSwapchain {
    pub unsafe fn new(
        window: &Window,
        instance: &Instance,
        surface: &vk::SurfaceKHR,
        rrdevice: &RRDevice,
    ) -> Self {
        let mut rrswapchain = create_swapchain(window, instance, surface, rrdevice).unwrap();
        let _ = create_swapchain_image_view(rrdevice, &mut rrswapchain);
        rrswapchain
    }
}

impl SwapchainSupport {
    pub unsafe fn get(
        instance: &Instance,
        surface: &vk::SurfaceKHR,
        physical_device: &vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(*physical_device, *surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(*physical_device, *surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(*physical_device, *surface)?,
        })
    }

    pub fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        formats
            .iter()
            .cloned()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or_else(|| formats[0])
    }

    pub fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        present_modes
            .iter()
            .cloned()
            .find(|m| *m == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO)
    }

    pub fn get_swapchain_extent(
        window: &Window,
        capabilities: vk::SurfaceCapabilitiesKHR,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            let size = window.inner_size();
            let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));
            vk::Extent2D::builder()
                .width(clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                    size.width,
                ))
                .height(clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                    size.height,
                ))
                .build()
        }
    }
}

pub unsafe fn create_swapchain(
    window: &Window,
    instance: &Instance,
    surface: &vk::SurfaceKHR,
    rrdevice: &RRDevice,
) -> Result<(RRSwapchain)> {
    let indices = QueueFamilyIndices::get(instance, surface, &rrdevice.physical_device)?;
    let support = SwapchainSupport::get(instance, surface, &rrdevice.physical_device)?;
    let surface_format = SwapchainSupport::get_swapchain_surface_format(&support.formats);
    let present_mode = SwapchainSupport::get_swapchain_present_mode(&support.present_modes);
    let extent = SwapchainSupport::get_swapchain_extent(window, support.capabilities);

    let mut image_count = support.capabilities.min_image_count + 1;
    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count
    {
        image_count = support.capabilities.max_image_count;
    }

    let mut queue_family_indices = vec![];
    let image_sharing_mode = if indices.graphics != indices.present {
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(*surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let mut rrswapchain = RRSwapchain::default();
    rrswapchain.swapchain = rrdevice.device.create_swapchain_khr(&info, None)?;
    rrswapchain.swapchain_images = rrdevice
        .device
        .get_swapchain_images_khr(rrswapchain.swapchain)?;
    rrswapchain.swapchain_format = surface_format.format;
    rrswapchain.swapchain_extent = extent;
    Ok((rrswapchain))
}

pub unsafe fn create_swapchain_image_view(
    rrdevice: &RRDevice,
    rrswapchain: &mut RRSwapchain,
) -> Result<()> {
    rrswapchain.swapchain_image_views = rrswapchain
        .swapchain_images
        .iter()
        .map(|i| {
            create_image_view(
                rrdevice,
                *i,
                rrswapchain.swapchain_format,
                vk::ImageAspectFlags::COLOR,
                1,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(())
}
