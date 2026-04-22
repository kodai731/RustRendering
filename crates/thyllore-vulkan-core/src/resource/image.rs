use crate::command::*;
use crate::core::device::*;
use crate::resource::buffer::*;
use crate::vulkan::*;
use std::fs::File;
use std::ptr::copy_nonoverlapping as memcpy;
use std::rc::Rc;

#[derive(Clone, Debug, Default)]
pub struct RRImage {
    pub image: vk::Image,
    pub image_memory: vk::DeviceMemory,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
}

impl RRImage {
    pub unsafe fn new(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrcommand_pool: &RRCommandPool,
    ) -> Result<Self> {
        let mut rrimage = RRImage::default();
        let (image, image_memory, mip_levels) =
            create_texture_image(instance, rrdevice, rrcommand_pool)?;
        let image_view = create_image_view(
            rrdevice,
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?;
        rrimage.image = image;
        rrimage.image_memory = image_memory;
        rrimage.image_view = image_view;
        let sampler = create_texture_sampler(&rrdevice, mip_levels)?;
        rrimage.sampler = sampler;
        Ok(rrimage)
    }

    pub unsafe fn new_from_pixels(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrcommand_pool: &Rc<RRCommandPool>,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (image, image_memory, mip_levels) = create_texture_image_pixel(
            instance,
            rrdevice,
            rrcommand_pool,
            &pixels.to_vec(),
            width,
            height,
        )?;

        let image_view = create_image_view(
            rrdevice,
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?;

        let sampler = create_texture_sampler(rrdevice, mip_levels)?;

        Ok(Self {
            image,
            image_memory,
            image_view,
            sampler,
        })
    }

    pub unsafe fn new_from_file(
        instance: &Instance,
        rrdevice: &RRDevice,
        rrcommand_pool: &RRCommandPool,
        file_path: &std::path::Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rrimage = RRImage::default();
        let (image, image_memory, mip_levels) =
            create_texture_image_from_file(instance, rrdevice, rrcommand_pool, file_path)?;

        let image_view = create_image_view(
            rrdevice,
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?;

        rrimage.image = image;
        rrimage.image_memory = image_memory;
        rrimage.image_view = image_view;

        let sampler = create_texture_sampler(&rrdevice, mip_levels)?;
        rrimage.sampler = sampler;

        Ok(rrimage)
    }

    pub unsafe fn destroy(&mut self, rrdevice: &RRDevice) {
        if self.sampler != vk::Sampler::null() {
            rrdevice.device.destroy_sampler(self.sampler, None);
            self.sampler = vk::Sampler::null();
        }
        if self.image_view != vk::ImageView::null() {
            rrdevice.device.destroy_image_view(self.image_view, None);
            self.image_view = vk::ImageView::null();
        }
        if self.image != vk::Image::null() {
            rrdevice.device.destroy_image(self.image, None);
            self.image = vk::Image::null();
        }
        if self.image_memory != vk::DeviceMemory::null() {
            rrdevice.device.free_memory(self.image_memory, None);
            self.image_memory = vk::DeviceMemory::null();
        }
    }
}

impl Drop for RRImage {
    fn drop(&mut self) {
        if self.image != vk::Image::null() {
            log_warn!("RRImage dropped without calling destroy()");
        }
    }
}

pub unsafe fn create_texture_image(
    instance: &Instance,
    rrdevice: &RRDevice,
    rrcommand_pool: &RRCommandPool,
) -> Result<(vk::Image, vk::DeviceMemory, u32)> {
    /*TODO :
     Try to experiment with this by creating a setup_command_buffer that the helper functions record commands into,
     and add a flush_setup_commands to execute the commands that have been recorded so far.
     It's best to do this after the texture mapping works to check if the texture resources are still set up correctly.
    */
    let image = File::open("assets/models/VikingRoom/viking_room.png")?;
    let decoder = png::Decoder::new(image);
    let mut reader = decoder.read_info()?;
    let mut pixels = vec![0; reader.info().raw_bytes()];
    reader.next_frame(&mut pixels)?;
    let size = reader.info().raw_bytes() as u64;
    let (width, height) = reader.info().size();
    let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

    if width != 1024 || height != 1024 || reader.info().color_type != png::ColorType::Rgba {
        return Err(anyhow!(
            "Invalid texture image: expected 1024x1024 RGBA, got {}x{} {:?}",
            width,
            height,
            reader.info().color_type
        ));
    }

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        rrdevice,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let memory =
        rrdevice
            .device
            .map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
    memcpy(pixels.as_ptr(), memory.cast(), pixels.len());
    rrdevice.device.unmap_memory(staging_buffer_memory);

    let (texture_image, texture_image_memory) = create_image(
        instance,
        rrdevice,
        width,
        height,
        mip_levels,
        vk::SampleCountFlags::_1,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    transition_image_layout(
        rrdevice,
        rrdevice.graphics_queue,
        rrcommand_pool.command_pool,
        texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        mip_levels,
    )?;
    copy_buffer_to_image(
        rrdevice,
        rrcommand_pool,
        staging_buffer,
        texture_image,
        width,
        height,
    )?;

    generate_mipmaps(
        instance,
        rrdevice,
        rrcommand_pool,
        texture_image,
        vk::Format::R8G8B8A8_SRGB,
        width,
        height,
        mip_levels,
    )?;

    rrdevice.device.destroy_buffer(staging_buffer, None);
    rrdevice.device.free_memory(staging_buffer_memory, None);

    Ok((texture_image, texture_image_memory, mip_levels))
}

pub unsafe fn create_texture_image_from_file(
    instance: &Instance,
    rrdevice: &RRDevice,
    rrcommand_pool: &RRCommandPool,
    file_path: &std::path::Path,
) -> Result<(vk::Image, vk::DeviceMemory, u32)> {
    let image = File::open(file_path)?;
    let decoder = png::Decoder::new(image);
    let mut reader = decoder.read_info()?;

    let info = reader.info();
    let (width, height) = info.size();
    let color_type = info.color_type;

    println!(
        "Loading texture: {:?}, size: {}x{}, color_type: {:?}",
        file_path, width, height, color_type
    );

    let mut pixels = vec![0; reader.info().raw_bytes()];
    reader.next_frame(&mut pixels)?;

    let rgba_pixels = match color_type {
        png::ColorType::Rgba => pixels,
        png::ColorType::Rgb => {
            let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
            for chunk in pixels.chunks(3) {
                rgba.push(chunk[0]);
                rgba.push(chunk[1]);
                rgba.push(chunk[2]);
                rgba.push(255);
            }
            rgba
        }
        png::ColorType::Grayscale => {
            let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
            for &gray in pixels.iter() {
                rgba.push(gray);
                rgba.push(gray);
                rgba.push(gray);
                rgba.push(255);
            }
            rgba
        }
        png::ColorType::GrayscaleAlpha => {
            let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
            for chunk in pixels.chunks(2) {
                let gray = chunk[0];
                let alpha = chunk[1];
                rgba.push(gray);
                rgba.push(gray);
                rgba.push(gray);
                rgba.push(alpha);
            }
            rgba
        }
        _ => {
            return Err(anyhow!("Unsupported color type: {:?}", color_type).into());
        }
    };

    let size = (rgba_pixels.len()) as u64;
    let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        rrdevice,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let memory =
        rrdevice
            .device
            .map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
    memcpy(rgba_pixels.as_ptr(), memory.cast(), rgba_pixels.len());
    rrdevice.device.unmap_memory(staging_buffer_memory);

    let (texture_image, texture_image_memory) = create_image(
        instance,
        rrdevice,
        width,
        height,
        mip_levels,
        vk::SampleCountFlags::_1,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    transition_image_layout(
        rrdevice,
        rrdevice.graphics_queue,
        rrcommand_pool.command_pool,
        texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        mip_levels,
    )?;
    copy_buffer_to_image(
        rrdevice,
        rrcommand_pool,
        staging_buffer,
        texture_image,
        width,
        height,
    )?;

    generate_mipmaps(
        instance,
        rrdevice,
        rrcommand_pool,
        texture_image,
        vk::Format::R8G8B8A8_SRGB,
        width,
        height,
        mip_levels,
    )?;

    rrdevice.device.destroy_buffer(staging_buffer, None);
    rrdevice.device.free_memory(staging_buffer_memory, None);

    Ok((texture_image, texture_image_memory, mip_levels))
}

pub unsafe fn create_image(
    instance: &Instance,
    rrdevice: &RRDevice,
    width: u32,
    height: u32,
    mip_levels: u32,
    samples: vk::SampleCountFlags,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(mip_levels)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED) // Not usable by the GPU and the very first transition will discard the texels.
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(samples)
        .flags(vk::ImageCreateFlags::empty());

    let image = rrdevice.device.create_image(&info, None)?;
    let requirements = rrdevice.device.get_image_memory_requirements(image);
    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            rrdevice.physical_device,
            properties,
            requirements,
        )?);
    let image_memory = rrdevice.device.allocate_memory(&info, None)?;
    rrdevice.device.bind_image_memory(image, image_memory, 0)?;

    Ok((image, image_memory))
}

pub unsafe fn transition_image_layout(
    rrdevice: &RRDevice,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    image: vk::Image,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    mip_levels: u32,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(rrdevice, command_pool)?;

    let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
        match format {
            vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT => {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            }
            _ => vk::ImageAspectFlags::DEPTH,
        }
    } else {
        vk::ImageAspectFlags::COLOR
    };

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspect_mask)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
        match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            ),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ),
            _ => return Err(anyhow!("Unsupported image layout transition")),
        };

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED) // barrier between queue families
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource)
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask);

    rrdevice.device.cmd_pipeline_barrier(
        command_buffer,
        src_stage_mask, // perations will wait on the barrier.
        dst_stage_mask, //
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_commands(rrdevice, queue, command_pool, command_buffer)?;

    Ok(())
}

// unsafe fn create_texture_image_view(device: &Device, data: &mut AppData) -> Result<()> {
//     data.texture_image_view = Self::create_image_view(
//         device,
//         data.texture_image,
//         vk::Format::R8G8B8A8_SRGB,
//         vk::ImageAspectFlags::COLOR,
//         data.mip_levels,
//     )?;
//
//     Ok(())
// }

pub unsafe fn create_image_view(
    rrdevice: &RRDevice,
    image: vk::Image,
    format: vk::Format,
    aspects: vk::ImageAspectFlags,
    mip_levels: u32,
) -> Result<vk::ImageView> {
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspects)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(format)
        .subresource_range(subresource_range);

    Ok(rrdevice.device.create_image_view(&info, None)?)
}

unsafe fn generate_mipmaps(
    instance: &Instance,
    rrdevice: &RRDevice,
    rrcommand_pool: &RRCommandPool,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<()> {
    if !instance
        .get_physical_device_format_properties(rrdevice.physical_device, format)
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    {
        return Err(anyhow!(
            "Texture image format does not system linear blitting"
        ));
    }

    let command_buffer = begin_single_time_commands(rrdevice, rrcommand_pool.command_pool)?;

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    let mut barrier = vk::ImageMemoryBarrier::builder()
        .image(image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(subresource);

    let mut mip_width = width;
    let mut mip_height = height;

    for i in 1..mip_levels {
        barrier.subresource_range.base_mip_level = i - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

        rrdevice.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i - 1)
            .base_array_layer(0)
            .layer_count(1);

        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i)
            .base_array_layer(0)
            .layer_count(1);

        let blit = vk::ImageBlit::builder()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width as i32,
                    y: mip_height as i32,
                    z: 1, // a 2D image has a depth of 1.
                },
            ])
            .src_subresource(src_subresource)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: (if mip_width > 1 { mip_width / 2 } else { 1 }) as i32,
                    y: (if mip_height > 1 { mip_height / 2 } else { 1 }) as i32,
                    z: 1,
                },
            ])
            .dst_subresource(dst_subresource);

        rrdevice.device.cmd_blit_image(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit],
            vk::Filter::LINEAR,
        );

        barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        rrdevice.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        if mip_width > 1 {
            mip_width /= 2;
        }
        if mip_height > 1 {
            mip_height /= 2;
        }
    }

    barrier.subresource_range.base_mip_level = mip_levels - 1;
    barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

    rrdevice.device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_commands(
        rrdevice,
        rrdevice.graphics_queue,
        rrcommand_pool.command_pool,
        command_buffer,
    )?;

    Ok(())
}

pub unsafe fn create_texture_sampler(
    rrdevice: &RRDevice,
    mip_levels: u32,
) -> Result<(vk::Sampler)> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(mip_levels as f32);
    let sampler = rrdevice.device.create_sampler(&info, None)?;

    Ok((sampler))
}

pub unsafe fn create_nearest_sampler(rrdevice: &RRDevice) -> Result<vk::Sampler> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .anisotropy_enable(false)
        .max_anisotropy(1.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(0.0);
    let sampler = rrdevice.device.create_sampler(&info, None)?;

    Ok(sampler)
}

pub unsafe fn create_texture_image_pixel(
    instance: &Instance,
    rrdevice: &RRDevice,
    rrcommand_pool: &Rc<RRCommandPool>,
    pixels: &Vec<u8>,
    width: u32,
    height: u32,
) -> Result<(vk::Image, vk::DeviceMemory, u32)> {
    /*TODO :
     Try to experiment with this by creating a setup_command_buffer that the helper functions record commands into,
     and add a flush_setup_commands to execute the commands that have been recorded so far.
     It's best to do this after the texture mapping works to check if the texture resources are still set up correctly.
    */
    let size = (size_of::<u8>() * pixels.len()) as u64;
    let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        rrdevice,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let memory =
        rrdevice
            .device
            .map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
    memcpy(pixels.as_ptr(), memory.cast(), pixels.len());
    rrdevice.device.unmap_memory(staging_buffer_memory);

    let (texture_image, texture_image_memory) = create_image(
        instance,
        rrdevice,
        width,
        height,
        mip_levels,
        vk::SampleCountFlags::_1,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    transition_image_layout(
        rrdevice,
        rrdevice.graphics_queue,
        rrcommand_pool.command_pool,
        texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        mip_levels,
    )?;
    copy_buffer_to_image(
        rrdevice,
        rrcommand_pool,
        staging_buffer,
        texture_image,
        width,
        height,
    )?;

    generate_mipmaps(
        instance,
        rrdevice,
        rrcommand_pool,
        texture_image,
        vk::Format::R8G8B8A8_SRGB,
        width,
        height,
        mip_levels,
    )?;

    rrdevice.device.destroy_buffer(staging_buffer, None);
    rrdevice.device.free_memory(staging_buffer_memory, None);

    Ok((texture_image, texture_image_memory, mip_levels))
}
