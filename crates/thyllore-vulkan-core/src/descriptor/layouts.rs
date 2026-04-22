use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::core::device::RRDevice;

#[derive(Clone, Debug, Default)]
pub struct DescriptorLayouts {
    pub frame: vk::DescriptorSetLayout,
    pub material: vk::DescriptorSetLayout,
    pub object: vk::DescriptorSetLayout,
}

impl DescriptorLayouts {
    pub unsafe fn new(rrdevice: &RRDevice) -> Result<Self> {
        let frame = Self::create_frame_layout(rrdevice)?;
        let material = Self::create_material_layout(rrdevice)?;
        let object = Self::create_object_layout(rrdevice)?;

        Ok(Self {
            frame,
            material,
            object,
        })
    }

    unsafe fn create_frame_layout(rrdevice: &RRDevice) -> Result<vk::DescriptorSetLayout> {
        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

        let bindings = &[ubo_binding];
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

        Ok(rrdevice.device.create_descriptor_set_layout(&info, None)?)
    }

    unsafe fn create_material_layout(rrdevice: &RRDevice) -> Result<vk::DescriptorSetLayout> {
        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = &[sampler_binding, ubo_binding];
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

        Ok(rrdevice.device.create_descriptor_set_layout(&info, None)?)
    }

    unsafe fn create_object_layout(rrdevice: &RRDevice) -> Result<vk::DescriptorSetLayout> {
        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let bindings = &[ubo_binding];
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

        Ok(rrdevice.device.create_descriptor_set_layout(&info, None)?)
    }

    pub fn as_array(&self) -> [vk::DescriptorSetLayout; 3] {
        [self.frame, self.material, self.object]
    }

    pub fn frame_object_array(&self) -> [vk::DescriptorSetLayout; 2] {
        [self.frame, self.object]
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        if self.frame != vk::DescriptorSetLayout::null() {
            device.destroy_descriptor_set_layout(self.frame, None);
            self.frame = vk::DescriptorSetLayout::null();
        }
        if self.material != vk::DescriptorSetLayout::null() {
            device.destroy_descriptor_set_layout(self.material, None);
            self.material = vk::DescriptorSetLayout::null();
        }
        if self.object != vk::DescriptorSetLayout::null() {
            device.destroy_descriptor_set_layout(self.object, None);
            self.object = vk::DescriptorSetLayout::null();
        }
    }
}

pub struct DescriptorAllocator {
    pools: Vec<vk::DescriptorPool>,
    current_pool_index: usize,
    sets_per_pool: u32,
    pool_sizes: Vec<vk::DescriptorPoolSize>,
}

impl DescriptorAllocator {
    pub unsafe fn new(
        rrdevice: &RRDevice,
        pool_sizes: Vec<vk::DescriptorPoolSize>,
        sets_per_pool: u32,
    ) -> Result<Self> {
        let pool = Self::create_pool(rrdevice, &pool_sizes, sets_per_pool)?;

        Ok(Self {
            pools: vec![pool],
            current_pool_index: 0,
            sets_per_pool,
            pool_sizes,
        })
    }

    unsafe fn create_pool(
        rrdevice: &RRDevice,
        pool_sizes: &[vk::DescriptorPoolSize],
        max_sets: u32,
    ) -> Result<vk::DescriptorPool> {
        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(max_sets)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        Ok(rrdevice.device.create_descriptor_pool(&info, None)?)
    }

    pub unsafe fn allocate(
        &mut self,
        rrdevice: &RRDevice,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        let layouts = &[layout];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.pools[self.current_pool_index])
            .set_layouts(layouts);

        match rrdevice.device.allocate_descriptor_sets(&info) {
            Ok(sets) => Ok(sets[0]),
            Err(vk::ErrorCode::OUT_OF_POOL_MEMORY | vk::ErrorCode::FRAGMENTED_POOL) => {
                let new_pool = Self::create_pool(rrdevice, &self.pool_sizes, self.sets_per_pool)?;
                self.pools.push(new_pool);
                self.current_pool_index = self.pools.len() - 1;

                let info = vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(self.pools[self.current_pool_index])
                    .set_layouts(layouts);

                let sets = rrdevice.device.allocate_descriptor_sets(&info)?;
                Ok(sets[0])
            }
            Err(e) => Err(e.into()),
        }
    }

    pub unsafe fn allocate_multiple(
        &mut self,
        rrdevice: &RRDevice,
        layout: vk::DescriptorSetLayout,
        count: usize,
    ) -> Result<Vec<vk::DescriptorSet>> {
        let layouts = vec![layout; count];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.pools[self.current_pool_index])
            .set_layouts(&layouts);

        match rrdevice.device.allocate_descriptor_sets(&info) {
            Ok(sets) => Ok(sets),
            Err(vk::ErrorCode::OUT_OF_POOL_MEMORY | vk::ErrorCode::FRAGMENTED_POOL) => {
                let new_pool = Self::create_pool(rrdevice, &self.pool_sizes, self.sets_per_pool)?;
                self.pools.push(new_pool);
                self.current_pool_index = self.pools.len() - 1;

                let info = vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(self.pools[self.current_pool_index])
                    .set_layouts(&layouts);

                let sets = rrdevice.device.allocate_descriptor_sets(&info)?;
                Ok(sets)
            }
            Err(e) => Err(e.into()),
        }
    }

    pub unsafe fn reset(&mut self, rrdevice: &RRDevice) -> Result<()> {
        for pool in &self.pools {
            rrdevice
                .device
                .reset_descriptor_pool(*pool, vk::DescriptorPoolResetFlags::empty())?;
        }
        self.current_pool_index = 0;
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        for pool in &self.pools {
            device.destroy_descriptor_pool(*pool, None);
        }
        self.pools.clear();
    }
}
