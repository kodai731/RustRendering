use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::core::RRDevice;
use crate::resource::hdr_buffer::HDR_FORMAT;
use crate::resource::render_target_transient::TransientDesc;

#[derive(Clone, Copy, Debug, Default)]
pub struct BloomMipTarget {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub framebuffer: vk::Framebuffer,
    pub extent: vk::Extent2D,
}

#[derive(Clone, Debug, Default)]
pub struct BloomChain {
    pub mip_extents: Vec<vk::Extent2D>,
    pub downsample_render_pass: vk::RenderPass,
    pub upsample_render_pass: vk::RenderPass,
    pub sampler: vk::Sampler,
}

impl BloomChain {
    pub unsafe fn new(
        rrdevice: &RRDevice,
        base_width: u32,
        base_height: u32,
        mip_count: u32,
    ) -> Result<Self> {
        let downsample_render_pass =
            Self::create_render_pass(rrdevice, vk::AttachmentLoadOp::DONT_CARE)?;
        let upsample_render_pass = Self::create_render_pass(rrdevice, vk::AttachmentLoadOp::LOAD)?;
        let sampler = Self::create_sampler(&rrdevice.device)?;

        let mip_extents = compute_mip_extents(base_width, base_height, mip_count);
        for (index, extent) in mip_extents.iter().enumerate() {
            log!("Bloom mip {}: {}x{}", index, extent.width, extent.height);
        }

        Ok(Self {
            mip_extents,
            downsample_render_pass,
            upsample_render_pass,
            sampler,
        })
    }

    pub fn mip_count(&self) -> usize {
        self.mip_extents.len()
    }

    pub fn mip_desc(&self, mip_index: usize) -> Option<TransientDesc> {
        self.mip_extents.get(mip_index).map(|extent| TransientDesc {
            width: extent.width,
            height: extent.height,
            format: HDR_FORMAT,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        })
    }

    unsafe fn create_render_pass(
        rrdevice: &RRDevice,
        load_op: vk::AttachmentLoadOp,
    ) -> Result<vk::RenderPass> {
        let initial_layout = if load_op == vk::AttachmentLoadOp::LOAD {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::UNDEFINED
        };

        let color_attachment = vk::AttachmentDescription::builder()
            .format(HDR_FORMAT)
            .samples(vk::SampleCountFlags::_1)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(initial_layout)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_attachments = [color_attachment_ref];

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachments);

        let dependency_in = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let dependency_out = vk::SubpassDependency::builder()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
            )
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        let attachments = [color_attachment];
        let subpasses = [subpass];
        let dependencies = [dependency_in, dependency_out];

        let info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = rrdevice.device.create_render_pass(&info, None)?;
        Ok(render_pass)
    }

    unsafe fn create_sampler(device: &vulkanalia::Device) -> Result<vk::Sampler> {
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(1.0);

        let sampler = device.create_sampler(&sampler_info, None)?;
        Ok(sampler)
    }

    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        let mip_count = self.mip_extents.len() as u32;
        self.mip_extents = compute_mip_extents(new_width, new_height, mip_count);
        log!("Resized bloom chain for {}x{}", new_width, new_height);
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        device.destroy_sampler(self.sampler, None);
        device.destroy_render_pass(self.downsample_render_pass, None);
        device.destroy_render_pass(self.upsample_render_pass, None);
        log!("Destroyed bloom chain");
    }
}

fn compute_mip_extents(base_width: u32, base_height: u32, mip_count: u32) -> Vec<vk::Extent2D> {
    let mut extents = Vec::with_capacity(mip_count as usize);
    let mut width = base_width / 2;
    let mut height = base_height / 2;

    for _ in 0..mip_count {
        width = width.max(1);
        height = height.max(1);
        extents.push(vk::Extent2D { width, height });
        width /= 2;
        height /= 2;
    }

    extents
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mip_extents_halve_and_clamp_to_one() {
        let extents = compute_mip_extents(16, 4, 5);

        let sizes: Vec<(u32, u32)> = extents.iter().map(|e| (e.width, e.height)).collect();
        assert_eq!(sizes, vec![(8, 2), (4, 1), (2, 1), (1, 1), (1, 1)]);
    }
}
