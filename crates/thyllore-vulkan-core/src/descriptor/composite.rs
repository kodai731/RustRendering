use crate::core::device::*;
use crate::vulkan::*;

pub const MAX_SELECTED_OBJECTS: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SelectionUBO {
    pub selected_ids: [[u32; 4]; MAX_SELECTED_OBJECTS],
    pub selected_count: u32,
    pub _padding: [u32; 3],
}

impl Default for SelectionUBO {
    fn default() -> Self {
        Self {
            selected_ids: [[0u32; 4]; MAX_SELECTED_OBJECTS],
            selected_count: 0,
            _padding: [0; 3],
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RRCompositeDescriptorSet {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub selection_buffer: vk::Buffer,
    pub selection_buffer_memory: vk::DeviceMemory,
}

impl RRCompositeDescriptorSet {
    /// Create Composite descriptor set layout
    pub unsafe fn create_layout(rrdevice: &RRDevice) -> Result<vk::DescriptorSetLayout> {
        // Binding 0: Position sampler (sampled image)
        let position_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        // Binding 1: Normal sampler (sampled image)
        let normal_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        // Binding 2: Shadow mask sampler (sampled image)
        let shadow_mask_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        // Binding 3: Albedo sampler (sampled image)
        let albedo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let scene_ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let object_id_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(5)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let selection_ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(6)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let bindings = [
            position_binding,
            normal_binding,
            shadow_mask_binding,
            albedo_binding,
            scene_ubo_binding,
            object_id_binding,
            selection_ubo_binding,
        ];

        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

        let layout = rrdevice.device.create_descriptor_set_layout(&info, None)?;

        Ok(layout)
    }

    pub unsafe fn create_pool(rrdevice: &RRDevice) -> Result<vk::DescriptorPool> {
        let sampler_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(5);

        let ubo_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(2);

        let pool_sizes = [sampler_size, ubo_size];

        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let pool = rrdevice.device.create_descriptor_pool(&info, None)?;

        Ok(pool)
    }

    pub unsafe fn allocate_and_update(
        &mut self,
        instance: &Instance,
        rrdevice: &RRDevice,
        position_image_view: vk::ImageView,
        position_sampler: vk::Sampler,
        normal_image_view: vk::ImageView,
        normal_sampler: vk::Sampler,
        shadow_mask_image_view: vk::ImageView,
        shadow_mask_sampler: vk::Sampler,
        albedo_image_view: vk::ImageView,
        albedo_sampler: vk::Sampler,
        scene_uniform_buffer: vk::Buffer,
        object_id_image_view: vk::ImageView,
        object_id_sampler: vk::Sampler,
    ) -> Result<()> {
        let layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = rrdevice.device.allocate_descriptor_sets(&alloc_info)?;
        self.descriptor_set = descriptor_sets[0];

        let (selection_buffer, selection_buffer_memory) =
            Self::create_selection_buffer(instance, rrdevice)?;
        self.selection_buffer = selection_buffer;
        self.selection_buffer_memory = selection_buffer_memory;

        let position_image_info = vk::DescriptorImageInfo::builder()
            .image_view(position_image_view)
            .sampler(position_sampler)
            .image_layout(vk::ImageLayout::GENERAL)
            .build();

        let position_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&position_image_info))
            .build();

        let normal_image_info = vk::DescriptorImageInfo::builder()
            .image_view(normal_image_view)
            .sampler(normal_sampler)
            .image_layout(vk::ImageLayout::GENERAL)
            .build();

        let normal_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&normal_image_info))
            .build();

        let shadow_mask_info = vk::DescriptorImageInfo::builder()
            .image_view(shadow_mask_image_view)
            .sampler(shadow_mask_sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let shadow_mask_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&shadow_mask_info))
            .build();

        let albedo_image_info = vk::DescriptorImageInfo::builder()
            .image_view(albedo_image_view)
            .sampler(albedo_sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let albedo_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(3)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&albedo_image_info))
            .build();

        let scene_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(scene_uniform_buffer)
            .offset(0)
            .range(std::mem::size_of::<crate::data::SceneUniformData>() as u64)
            .build();

        let scene_ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(4)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&scene_buffer_info))
            .build();

        let object_id_image_info = vk::DescriptorImageInfo::builder()
            .image_view(object_id_image_view)
            .sampler(object_id_sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let object_id_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(5)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&object_id_image_info))
            .build();

        let selection_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(self.selection_buffer)
            .offset(0)
            .range(std::mem::size_of::<SelectionUBO>() as u64)
            .build();

        let selection_ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(6)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&selection_buffer_info))
            .build();

        let writes = [
            position_write,
            normal_write,
            shadow_mask_write,
            albedo_write,
            scene_ubo_write,
            object_id_write,
            selection_ubo_write,
        ];

        rrdevice
            .device
            .update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);

        Ok(())
    }

    pub unsafe fn update_gbuffer_views(
        &self,
        rrdevice: &RRDevice,
        position_image_view: vk::ImageView,
        position_sampler: vk::Sampler,
        normal_image_view: vk::ImageView,
        normal_sampler: vk::Sampler,
        shadow_mask_image_view: vk::ImageView,
        shadow_mask_sampler: vk::Sampler,
        albedo_image_view: vk::ImageView,
        albedo_sampler: vk::Sampler,
        object_id_image_view: vk::ImageView,
        object_id_sampler: vk::Sampler,
    ) {
        if self.descriptor_set == vk::DescriptorSet::null() {
            return;
        }

        let position_image_info = vk::DescriptorImageInfo::builder()
            .image_view(position_image_view)
            .sampler(position_sampler)
            .image_layout(vk::ImageLayout::GENERAL)
            .build();

        let position_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&position_image_info))
            .build();

        let normal_image_info = vk::DescriptorImageInfo::builder()
            .image_view(normal_image_view)
            .sampler(normal_sampler)
            .image_layout(vk::ImageLayout::GENERAL)
            .build();

        let normal_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&normal_image_info))
            .build();

        let shadow_mask_info = vk::DescriptorImageInfo::builder()
            .image_view(shadow_mask_image_view)
            .sampler(shadow_mask_sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let shadow_mask_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&shadow_mask_info))
            .build();

        let albedo_image_info = vk::DescriptorImageInfo::builder()
            .image_view(albedo_image_view)
            .sampler(albedo_sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let albedo_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(3)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&albedo_image_info))
            .build();

        let object_id_image_info = vk::DescriptorImageInfo::builder()
            .image_view(object_id_image_view)
            .sampler(object_id_sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let object_id_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(5)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&object_id_image_info))
            .build();

        let writes = [
            position_write,
            normal_write,
            shadow_mask_write,
            albedo_write,
            object_id_write,
        ];

        rrdevice
            .device
            .update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
    }

    unsafe fn create_selection_buffer(
        instance: &Instance,
        rrdevice: &RRDevice,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        use crate::resource::buffer::create_buffer;

        let buffer_size = std::mem::size_of::<SelectionUBO>() as u64;

        let (buffer, memory) = create_buffer(
            instance,
            rrdevice,
            buffer_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        Ok((buffer, memory))
    }

    pub unsafe fn update_selection(
        &self,
        rrdevice: &RRDevice,
        selected_mesh_ids: &[u32],
    ) -> Result<()> {
        let mut ubo = SelectionUBO::default();
        let count = selected_mesh_ids.len().min(MAX_SELECTED_OBJECTS);

        for (i, &id) in selected_mesh_ids.iter().take(count).enumerate() {
            ubo.selected_ids[i] = [id, 0, 0, 0];
        }
        ubo.selected_count = count as u32;

        let memory = rrdevice.device.map_memory(
            self.selection_buffer_memory,
            0,
            std::mem::size_of::<SelectionUBO>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        std::ptr::copy_nonoverlapping(&ubo, memory as *mut SelectionUBO, 1);

        rrdevice.device.unmap_memory(self.selection_buffer_memory);

        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        if self.selection_buffer != vk::Buffer::null() {
            device.destroy_buffer(self.selection_buffer, None);
            self.selection_buffer = vk::Buffer::null();
        }

        if self.selection_buffer_memory != vk::DeviceMemory::null() {
            device.free_memory(self.selection_buffer_memory, None);
            self.selection_buffer_memory = vk::DeviceMemory::null();
        }

        if self.descriptor_pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.descriptor_pool = vk::DescriptorPool::null();
        }

        if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.descriptor_set_layout = vk::DescriptorSetLayout::null();
        }
    }
}
