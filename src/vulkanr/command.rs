use super::device::*;
use super::vulkan::*;
use crate::vulkanr::buffer::{RRBuffer, RRIndexBuffer, RRVertexBuffer};
use crate::vulkanr::descriptor::RRDescriptorSet;
use crate::vulkanr::pipeline::RRPipeline;
use crate::vulkanr::render::RRRender;
use crate::vulkanr::swapchain::RRSwapchain;
use glutin::surface::Surface;
use std::rc::Rc;
use vulkanalia::vk::Pipeline;

pub unsafe fn begin_single_time_commands(
    rrdevice: &RRDevice,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer> {
    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);
    let command_buffer = rrdevice.device.allocate_command_buffers(&info)?[0];

    let info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    rrdevice
        .device
        .begin_command_buffer(command_buffer, &info)?;

    Ok(command_buffer)
}

pub unsafe fn end_single_time_commands(
    rrdevice: &RRDevice,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    rrdevice.device.end_command_buffer(command_buffer)?;

    let command_buffers = &[command_buffer];
    let info = vk::SubmitInfo::builder().command_buffers(command_buffers);
    rrdevice
        .device
        .queue_submit(queue, &[info], vk::Fence::null())?;
    rrdevice.device.queue_wait_idle(queue)?;

    rrdevice
        .device
        .free_command_buffers(command_pool, &[command_buffer]);

    Ok(())
}

#[derive(Clone, Debug, Default)]
pub struct RRCommandPool {
    pub command_pool: vk::CommandPool, // Command pools manage the memory that is used to store the buffers
}

impl RRCommandPool {
    pub unsafe fn new(instance: &Instance, surface: &vk::SurfaceKHR, rrdevice: &RRDevice) -> Self {
        let mut rrcommand_pool = RRCommandPool::default();
        if let Err(e) = create_command_pool(instance, surface, rrdevice, &mut rrcommand_pool) {
            eprintln!("Create command pool failed {:?}", e);
        }
        println!("Created command pool");
        rrcommand_pool
    }
}

#[derive(Clone, Debug, Default)]
pub struct RRCommandBuffer {
    pub rrcommand_pool: Rc<RRCommandPool>,
    pub command_buffers: Vec<vk::CommandBuffer>,
}

impl RRCommandBuffer {
    pub unsafe fn new(rrcommand_pool: &Rc<RRCommandPool>) -> Self {
        let mut rrcommand_buffer = RRCommandBuffer::default();
        rrcommand_buffer.rrcommand_pool = Rc::clone(rrcommand_pool);
        rrcommand_buffer
    }
}

unsafe fn create_command_pool(
    instance: &Instance,
    surface: &vk::SurfaceKHR,
    rrdevice: &RRDevice,
    rrcommand_buffer: &mut RRCommandPool,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, surface, &rrdevice.physical_device)?;
    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::empty())
        .queue_family_index(indices.graphics); // Each command pool can only allocate command buffers that are submitted on a single type of queue.

    rrcommand_buffer.command_pool = rrdevice.device.create_command_pool(&info, None)?;

    Ok(())
}

// TODO: パイプラインのデータをクラスにまとめる
pub unsafe fn create_command_buffers(
    rrdevice: &RRDevice,
    rrrender: &RRRender,
    rrswapchain: &RRSwapchain,
    grid_pipeline: &RRPipeline,
    grid_descriptor_set: &RRDescriptorSet,
    grid_vertex_buffer: &RRVertexBuffer,
    grid_index_buffer: &RRIndexBuffer,
    model_pipeline: &RRPipeline,
    model_descriptor_set: &RRDescriptorSet,
    model_vertex_buffer: &RRVertexBuffer,
    model_index_buffer: &RRIndexBuffer,
    rrcommand_buffer: &mut RRCommandBuffer,
) -> Result<()> {
    let info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(rrcommand_buffer.rrcommand_pool.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(rrrender.framebuffers.len() as u32);
    rrcommand_buffer.command_buffers = rrdevice.device.allocate_command_buffers(&info)?;

    for (i, command_buffer) in rrcommand_buffer.command_buffers.iter().enumerate() {
        let inheritance = vk::CommandBufferInheritanceInfo::builder(); //  only relevant for secondary command buffers.
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::empty())
            .inheritance_info(&inheritance);

        rrdevice
            .device
            .begin_command_buffer(*command_buffer, &begin_info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(rrswapchain.swapchain_extent);
        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0, // The range of depths in the depth buffer is 0.0 to 1.0 in Vulkan, where 1.0 lies at the far view plane and 0.0 at the near view plane
                stencil: 0,
            },
        };

        let clear_values = &[color_clear_value, depth_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(rrrender.render_pass)
            .framebuffer(rrrender.framebuffers[i])
            .render_area(render_area)
            .clear_values(clear_values);
        rrdevice
            .device
            .cmd_begin_render_pass(*command_buffer, &info, vk::SubpassContents::INLINE);

        // grid
        rrdevice.device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            grid_pipeline.pipeline,
        );
        rrdevice.device.cmd_bind_vertex_buffers(
            *command_buffer,
            0,
            &[grid_vertex_buffer.buffer],
            &[0],
        );
        rrdevice.device.cmd_bind_index_buffer(
            *command_buffer,
            grid_index_buffer.buffer,
            0,
            vk::IndexType::UINT32,
        );
        rrdevice.device.cmd_bind_descriptor_sets(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            grid_pipeline.pipeline_layout,
            0,
            &[grid_descriptor_set.descriptor_sets[i]],
            &[],
        );
        rrdevice
            .device
            .cmd_draw_indexed(*command_buffer, grid_index_buffer.indices, 1, 0, 0, 0);

        // model
        rrdevice.device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            model_pipeline.pipeline,
        );
        rrdevice.device.cmd_bind_vertex_buffers(
            *command_buffer,
            0,
            &[model_vertex_buffer.buffer],
            &[0],
        );
        rrdevice.device.cmd_bind_index_buffer(
            *command_buffer,
            model_index_buffer.buffer,
            0,
            vk::IndexType::UINT32,
        );
        rrdevice.device.cmd_bind_descriptor_sets(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            model_pipeline.pipeline_layout,
            0,
            &[model_descriptor_set.descriptor_sets[i]],
            &[],
        );
        rrdevice
            .device
            .cmd_draw_indexed(*command_buffer, model_index_buffer.indices, 1, 0, 0, 0);

        rrdevice.device.cmd_end_render_pass(*command_buffer);
        rrdevice.device.end_command_buffer(*command_buffer)?;
    }

    Ok(())
}
