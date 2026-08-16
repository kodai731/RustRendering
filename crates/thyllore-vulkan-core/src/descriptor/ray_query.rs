use crate::core::device::*;
use crate::vulkan::*;

#[derive(Clone, Debug, Default)]
pub struct RRRayQueryDescriptorSet {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
}

impl RRRayQueryDescriptorSet {
    pub fn layout_bindings() -> Vec<vk::DescriptorSetLayoutBinding> {
        let position_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build();

        let normal_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build();

        let shadow_mask_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build();

        let tlas_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build();

        let scene_ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build();

        vec![
            position_binding,
            normal_binding,
            shadow_mask_binding,
            tlas_binding,
            scene_ubo_binding,
        ]
    }

    pub unsafe fn create_layout(rrdevice: &RRDevice) -> Result<vk::DescriptorSetLayout> {
        let bindings = Self::layout_bindings();
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        Ok(rrdevice.device.create_descriptor_set_layout(&info, None)?)
    }

    pub unsafe fn create_pool(rrdevice: &RRDevice) -> Result<vk::DescriptorPool> {
        let storage_image_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(3); // position, normal, shadow mask

        let accel_struct_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1);

        let ubo_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1);

        let pool_sizes = [storage_image_size, accel_struct_size, ubo_size];

        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let pool = rrdevice.device.create_descriptor_pool(&info, None)?;
        log::info!("Created Ray Query descriptor pool");

        Ok(pool)
    }

    pub unsafe fn allocate_and_update(
        &mut self,
        rrdevice: &RRDevice,
        position_image_view: vk::ImageView,
        normal_image_view: vk::ImageView,
        shadow_mask_image_view: vk::ImageView,
        tlas: vk::AccelerationStructureKHR,
        scene_uniform_buffer: vk::Buffer,
    ) -> Result<()> {
        let layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = rrdevice.device.allocate_descriptor_sets(&alloc_info)?;
        self.descriptor_set = descriptor_sets[0];

        let position_image_info = vk::DescriptorImageInfo::builder()
            .image_view(position_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .build();

        let position_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&position_image_info))
            .build();

        let normal_image_info = vk::DescriptorImageInfo::builder()
            .image_view(normal_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .build();

        let normal_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&normal_image_info))
            .build();

        let shadow_mask_info = vk::DescriptorImageInfo::builder()
            .image_view(shadow_mask_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .build();

        let shadow_mask_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&shadow_mask_info))
            .build();

        let tlas_handles = [tlas];
        let mut tlas_info = vk::WriteDescriptorSetAccelerationStructureKHR::builder()
            .acceleration_structures(&tlas_handles);

        let mut tlas_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(3)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut tlas_info)
            .build();
        tlas_write.descriptor_count = 1;

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

        let writes = [
            position_write,
            normal_write,
            shadow_mask_write,
            tlas_write,
            scene_ubo_write,
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
        normal_image_view: vk::ImageView,
        shadow_mask_image_view: vk::ImageView,
    ) {
        if self.descriptor_set == vk::DescriptorSet::null() {
            return;
        }

        let position_image_info = vk::DescriptorImageInfo::builder()
            .image_view(position_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .build();

        let position_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&position_image_info))
            .build();

        let normal_image_info = vk::DescriptorImageInfo::builder()
            .image_view(normal_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .build();

        let normal_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&normal_image_info))
            .build();

        let shadow_mask_info = vk::DescriptorImageInfo::builder()
            .image_view(shadow_mask_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .build();

        let shadow_mask_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&shadow_mask_info))
            .build();

        let writes = [position_write, normal_write, shadow_mask_write];

        rrdevice
            .device
            .update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
    }

    pub unsafe fn update_tlas(
        &mut self,
        rrdevice: &RRDevice,
        tlas: vk::AccelerationStructureKHR,
    ) -> Result<()> {
        if self.descriptor_set == vk::DescriptorSet::null() {
            return Ok(());
        }

        let tlas_handles = [tlas];
        let mut tlas_info = vk::WriteDescriptorSetAccelerationStructureKHR::builder()
            .acceleration_structures(&tlas_handles);

        let mut tlas_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(3)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut tlas_info)
            .build();
        tlas_write.descriptor_count = 1;

        rrdevice
            .device
            .update_descriptor_sets(&[tlas_write], &[] as &[vk::CopyDescriptorSet]);

        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
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
