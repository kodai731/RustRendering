use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_0::*;

const INITIAL_SETS_PER_POOL: u32 = 16;
const MAX_SETS_PER_POOL: u32 = 512;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct PoolSignature {
    descriptors_per_set: Vec<(i32, u32)>,
}

impl PoolSignature {
    pub fn from_bindings(bindings: &[vk::DescriptorSetLayoutBinding]) -> Self {
        let mut counts: BTreeMap<i32, u32> = BTreeMap::new();
        for binding in bindings {
            *counts.entry(binding.descriptor_type.as_raw()).or_default() +=
                binding.descriptor_count;
        }
        Self {
            descriptors_per_set: counts.into_iter().collect(),
        }
    }

    fn pool_sizes(&self, max_sets: u32) -> Vec<vk::DescriptorPoolSize> {
        self.descriptors_per_set
            .iter()
            .map(|&(raw_type, count)| {
                vk::DescriptorPoolSize::builder()
                    .type_(vk::DescriptorType::from_raw(raw_type))
                    .descriptor_count(count * max_sets)
                    .build()
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
struct PoolBlock {
    pool: vk::DescriptorPool,
    capacity: u32,
    used: u32,
}

impl PoolBlock {
    fn remaining(&self) -> u32 {
        self.capacity - self.used
    }
}

#[derive(Clone, Debug, Default)]
pub struct DescriptorAllocator {
    pools: BTreeMap<PoolSignature, Vec<PoolBlock>>,
}

impl DescriptorAllocator {
    pub unsafe fn allocate(
        &mut self,
        device: &Device,
        layout: vk::DescriptorSetLayout,
        signature: &PoolSignature,
        count: usize,
    ) -> Result<Vec<vk::DescriptorSet>> {
        let mut sets = Vec::with_capacity(count);
        let mut remaining = count as u32;

        while remaining > 0 {
            let block = self.block_with_room(device, signature)?;
            let batch = remaining.min(block.remaining());
            let layouts = vec![layout; batch as usize];
            let info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(block.pool)
                .set_layouts(&layouts);
            sets.extend(device.allocate_descriptor_sets(&info)?);
            block.used += batch;
            remaining -= batch;
        }

        Ok(sets)
    }

    unsafe fn block_with_room(
        &mut self,
        device: &Device,
        signature: &PoolSignature,
    ) -> Result<&mut PoolBlock> {
        let blocks = self.pools.entry(signature.clone()).or_default();
        let has_room = blocks.last().is_some_and(|block| block.remaining() > 0);
        if !has_room {
            let capacity = blocks.last().map_or(INITIAL_SETS_PER_POOL, |block| {
                (block.capacity * 2).min(MAX_SETS_PER_POOL)
            });
            let pool_sizes = signature.pool_sizes(capacity);
            let info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&pool_sizes)
                .max_sets(capacity);
            let pool = device.create_descriptor_pool(&info, None)?;
            blocks.push(PoolBlock {
                pool,
                capacity,
                used: 0,
            });
        }
        blocks
            .last_mut()
            .ok_or_else(|| anyhow!("descriptor pool block missing after creation"))
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        for block in self.pools.values().flatten() {
            device.destroy_descriptor_pool(block.pool, None);
        }
        self.pools.clear();
    }
}
