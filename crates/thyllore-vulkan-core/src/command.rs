use super::device::*;
use super::vulkan::*;
use crate::core::swapchain::RRSwapchain;
use crate::descriptor::model::RRDescriptorSet;
use crate::pipeline::RRPipeline;
use crate::render::pass::RRRender;
use crate::resource::buffer::{RRIndexBuffer, RRVertexBuffer};
use crate::resource::graphics_resource::GraphicsResources;
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

    // TODO: summarize data in pipeline
    pub unsafe fn bind_command(
        rrdevice: &RRDevice,
        rrrender: &RRRender,
        rrswapchain: &RRSwapchain,
        rrbind_info: &Vec<RRBindInfo>,
        rrcommand_buffer: &mut RRCommandBuffer,
        frame_index: usize,
    ) -> Result<()> {
        let inheritance = vk::CommandBufferInheritanceInfo::builder(); //  only relevant for secondary command buffers.
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::empty())
            .inheritance_info(&inheritance);
        let command_buffer = rrcommand_buffer.command_buffers[frame_index];

        rrdevice
            .device
            .begin_command_buffer(command_buffer, &begin_info)?;

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
                depth: 0.0,
                stencil: 0,
            },
        };

        let clear_values = &[color_clear_value, depth_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(rrrender.render_pass)
            .framebuffer(rrrender.framebuffers[frame_index])
            .render_area(render_area)
            .clear_values(clear_values);
        rrdevice
            .device
            .cmd_begin_render_pass(command_buffer, &info, vk::SubpassContents::INLINE);

        for i in 0..rrbind_info.len() {
            let bind_info = &rrbind_info[i];
            rrdevice.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                bind_info.rrpipeline.pipeline,
            );

            rrdevice.device.cmd_set_line_width(command_buffer, 1.0);

            rrdevice.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[bind_info.rrvertex_buffer.buffer],
                &[0],
            );
            rrdevice.device.cmd_bind_index_buffer(
                command_buffer,
                bind_info.rrindex_buffer.buffer,
                0,
                vk::IndexType::UINT32,
            );

            if let Some(graphics_resources) = bind_info.graphics_resources {
                let frame_set = graphics_resources.frame_set.sets[frame_index];
                let object_set_idx = graphics_resources
                    .objects
                    .get_set_index(frame_index, bind_info.object_index);
                let object_set = graphics_resources.objects.sets[object_set_idx];

                rrdevice.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    bind_info.rrpipeline.pipeline_layout,
                    0,
                    &[frame_set],
                    &[],
                );

                if let Some(material_id) = bind_info.material_id {
                    if let Some(material) = graphics_resources.materials.get(material_id) {
                        rrdevice.device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            bind_info.rrpipeline.pipeline_layout,
                            1,
                            &[material.descriptor_set],
                            &[],
                        );
                    }
                }

                rrdevice.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    bind_info.rrpipeline.pipeline_layout,
                    2,
                    &[object_set],
                    &[],
                );
            } else {
                let swapchain_images_len = bind_info.rrdescriptor_set.descriptor_sets.len()
                    / bind_info.rrdescriptor_set.rrdata.len().max(1);
                let descriptor_set_index =
                    bind_info.data_index * swapchain_images_len + frame_index;
                rrdevice.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    bind_info.rrpipeline.pipeline_layout,
                    0,
                    &[bind_info.rrdescriptor_set.descriptor_sets[descriptor_set_index]],
                    &[],
                );
            }

            rrdevice.device.cmd_draw_indexed(
                command_buffer,
                bind_info.rrindex_buffer.indices,
                1,
                bind_info.offset_index,
                bind_info.offset_index as i32,
                0,
            );
        }

        rrdevice.device.cmd_end_render_pass(command_buffer);
        rrdevice.device.end_command_buffer(command_buffer)?;

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

use crate::resource::graphics_resource::MaterialId;

#[derive(Clone, Debug)]
pub struct RRBindInfo<'a> {
    pub rrpipeline: &'a RRPipeline,
    pub rrdescriptor_set: &'a RRDescriptorSet,
    pub rrvertex_buffer: &'a RRVertexBuffer,
    pub rrindex_buffer: &'a RRIndexBuffer,
    pub offset_vertex: u32,
    pub offset_index: u32,
    pub data_index: usize,
    pub graphics_resources: Option<&'a GraphicsResources>,
    pub object_index: usize,
    pub material_id: Option<MaterialId>,
}

impl<'a> RRBindInfo<'a> {
    pub unsafe fn new(
        rrpipeline: &'a RRPipeline,
        rrdescriptor_set: &'a RRDescriptorSet,
        rrvertex_buffer: &'a RRVertexBuffer,
        rrindex_buffer: &'a RRIndexBuffer,
        offset_vertex: u32,
        offset_index: u32,
        data_index: usize,
    ) -> Self {
        Self {
            rrpipeline,
            rrdescriptor_set,
            rrvertex_buffer,
            rrindex_buffer,
            offset_vertex,
            offset_index,
            data_index,
            graphics_resources: None,
            object_index: 0,
            material_id: None,
        }
    }

    pub unsafe fn with_graphics_resources(
        rrpipeline: &'a RRPipeline,
        rrdescriptor_set: &'a RRDescriptorSet,
        rrvertex_buffer: &'a RRVertexBuffer,
        rrindex_buffer: &'a RRIndexBuffer,
        graphics_resources: &'a GraphicsResources,
        object_index: usize,
        material_id: MaterialId,
    ) -> Self {
        Self {
            rrpipeline,
            rrdescriptor_set,
            rrvertex_buffer,
            rrindex_buffer,
            offset_vertex: 0,
            offset_index: 0,
            data_index: 0,
            graphics_resources: Some(graphics_resources),
            object_index,
            material_id: Some(material_id),
        }
    }

    pub unsafe fn with_graphics_resources_no_material(
        rrpipeline: &'a RRPipeline,
        rrdescriptor_set: &'a RRDescriptorSet,
        rrvertex_buffer: &'a RRVertexBuffer,
        rrindex_buffer: &'a RRIndexBuffer,
        graphics_resources: &'a GraphicsResources,
        object_index: usize,
    ) -> Self {
        Self {
            rrpipeline,
            rrdescriptor_set,
            rrvertex_buffer,
            rrindex_buffer,
            offset_vertex: 0,
            offset_index: 0,
            data_index: 0,
            graphics_resources: Some(graphics_resources),
            object_index,
            material_id: None,
        }
    }
}
