use super::device::*;
use super::vulkan::*;
use crate::render::pass::RRRender;
use std::rc::Rc;

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

    // grid, grid, model, model
    pub unsafe fn allocate_command_buffers(
        rrdevice: &RRDevice,
        rrrender: &RRRender,
        rrcommand_buffer: &mut RRCommandBuffer,
    ) -> Result<()> {
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(rrcommand_buffer.rrcommand_pool.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count((rrrender.framebuffers.len() * 2) as u32);
        rrcommand_buffer.command_buffers = rrdevice.device.allocate_command_buffers(&info)?;
        Ok(())
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
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(indices.graphics); // Each command pool can only allocate command buffers that are submitted on a single type of queue.

    rrcommand_buffer.command_pool = rrdevice.device.create_command_pool(&info, None)?;

    Ok(())
}
