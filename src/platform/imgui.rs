use vulkanalia::prelude::v1_0::*;

use crate::app::init::MAX_FRAMES_IN_FLIGHT;
use crate::vulkanr::descriptor::ReflectedSetLayout;

#[derive(Clone, Debug)]
pub struct ImguiData {
    pub pipeline: Option<vk::Pipeline>,
    pub pipeline_layout: Option<vk::PipelineLayout>,
    pub descriptor_set: Option<vk::DescriptorSet>,
    pub descriptor_set_layout: Option<ReflectedSetLayout>,
    pub descriptor_pool: Option<vk::DescriptorPool>,
    pub font_image: Option<vk::Image>,
    pub font_image_memory: Option<vk::DeviceMemory>,
    pub font_image_view: Option<vk::ImageView>,
    pub sampler: Option<vk::Sampler>,
    pub vertex_buffers: [Option<vk::Buffer>; MAX_FRAMES_IN_FLIGHT],
    pub vertex_buffer_memories: [Option<vk::DeviceMemory>; MAX_FRAMES_IN_FLIGHT],
    pub vertex_buffer_sizes: [vk::DeviceSize; MAX_FRAMES_IN_FLIGHT],
    pub index_buffers: [Option<vk::Buffer>; MAX_FRAMES_IN_FLIGHT],
    pub index_buffer_memories: [Option<vk::DeviceMemory>; MAX_FRAMES_IN_FLIGHT],
    pub index_buffer_sizes: [vk::DeviceSize; MAX_FRAMES_IN_FLIGHT],
}

impl Default for ImguiData {
    fn default() -> Self {
        Self {
            pipeline: None,
            pipeline_layout: None,
            descriptor_set: None,
            descriptor_set_layout: None,
            descriptor_pool: None,
            font_image: None,
            font_image_memory: None,
            font_image_view: None,
            sampler: None,
            vertex_buffers: [None, None],
            vertex_buffer_memories: [None, None],
            vertex_buffer_sizes: [0, 0],
            index_buffers: [None, None],
            index_buffer_memories: [None, None],
            index_buffer_sizes: [0, 0],
        }
    }
}
