use vulkanalia::prelude::v1_0::*;

/// Converts raw query ticks to milliseconds.
const TIMESTAMP_QUERY_COUNT: u32 = 128;

pub fn ticks_to_ms(start_tick: u64, end_tick: u64, timestamp_period: f32) -> f32 {
    (end_tick.saturating_sub(start_tick)) as f32 * timestamp_period / 1e6
}

/// Decodes query results from the WITH_AVAILABILITY layout.
pub fn decode_results(
    labels: &[(String, u32, u32)], // (label, start_idx, end_idx)
    data: &[u64],
) -> Option<Vec<(String, u64, u64)>> {
    let mut results = Vec::new();
    for (label, start_idx, end_idx) in labels {
        let si = *start_idx as usize;
        let ei = *end_idx as usize;
        if 2 * ei + 1 >= data.len() {
            return None;
        }
        if data[2 * si + 1] == 0 || data[2 * ei + 1] == 0 {
            return None;
        }
        results.push((label.clone(), data[2 * si], data[2 * ei]));
    }
    Some(results)
}

pub struct GpuTimestampProfiler {
    pools: Vec<vk::QueryPool>,
    labels: Vec<Vec<(String, u32, u32)>>,
    next_query: Vec<u32>,
    timestamp_period: f32,
}

impl GpuTimestampProfiler {
    pub unsafe fn new(
        device: &vulkanalia::Device,
        timestamp_period: f32,
        frames_in_flight: usize,
    ) -> Self {
        let mut pools = Vec::with_capacity(frames_in_flight);
        let mut labels = Vec::with_capacity(frames_in_flight);
        let mut next_query = Vec::with_capacity(frames_in_flight);

        for _ in 0..frames_in_flight {
            let pool_info = vk::QueryPoolCreateInfo::builder()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(TIMESTAMP_QUERY_COUNT);
            let pool = device.create_query_pool(&pool_info, None).unwrap();
            pools.push(pool);
            labels.push(Vec::new());
            next_query.push(0);
        }

        Self {
            pools,
            labels,
            next_query,
            timestamp_period,
        }
    }

    pub unsafe fn begin_frame(
        &mut self,
        device: &vulkanalia::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
    ) {
        device.cmd_reset_query_pool(cmd, self.pools[slot], 0, TIMESTAMP_QUERY_COUNT);
        self.labels[slot].clear();
        self.next_query[slot] = 0;
    }

    pub unsafe fn begin_scope(
        &mut self,
        device: &vulkanalia::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
        label: String,
    ) {
        let q = self.next_query[slot];
        if q >= TIMESTAMP_QUERY_COUNT {
            return;
        }
        device.cmd_write_timestamp(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            self.pools[slot],
            q,
        );
        self.next_query[slot] += 1;
        self.labels[slot].push((label, q, q));
    }

    pub unsafe fn end_scope(
        &mut self,
        device: &vulkanalia::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
    ) {
        let pending = match self.labels[slot].last_mut() {
            Some(entry) => entry,
            None => return,
        };
        let q = self.next_query[slot];
        if q >= TIMESTAMP_QUERY_COUNT {
            return;
        }
        device.cmd_write_timestamp(
            cmd,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            self.pools[slot],
            q,
        );
        pending.2 = q;
        self.next_query[slot] += 1;
    }

    pub unsafe fn collect(
        &mut self,
        device: &vulkanalia::Device,
        slot: usize,
    ) -> Option<Vec<(String, f32)>> {
        let used = self.next_query[slot];
        if used == 0 {
            return Some(Vec::new());
        }

        let word_count = used * 2;
        let mut data = vec![0u8; (word_count as usize) * 8];

        let result = device.get_query_pool_results(
            self.pools[slot],
            0,
            used,
            &mut data,
            16,
            vk::QueryResultFlags::_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
        );

        match result {
            Ok(code) if code == vk::SuccessCode::NOT_READY => return None,
            Err(_) => return None,
            Ok(_) => {}
        }

        let words: Vec<u64> = data
            .chunks_exact(8)
            .map(|c| u64::from_ne_bytes(c.try_into().unwrap()))
            .collect::<Vec<u64>>();
        let decoded = decode_results(&self.labels[slot], &words)?;
        let results: Vec<(String, f32)> = decoded
            .into_iter()
            .map(|(label, start_tick, end_tick)| {
                (
                    label,
                    ticks_to_ms(start_tick, end_tick, self.timestamp_period),
                )
            })
            .collect();
        Some(results)
    }

    pub unsafe fn destroy(&mut self, device: &vulkanalia::Device) {
        for pool in &mut self.pools {
            if *pool != vk::QueryPool::null() {
                device.destroy_query_pool(*pool, None);
                *pool = vk::QueryPool::null();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ticks_to_ms() {
        assert_eq!(ticks_to_ms(100, 200, 1000000.0), 100.0);
    }

    #[test]
    fn test_decode_unavailable() {
        let labels = vec![("Scope".to_string(), 0, 1)];
        // data[0]=value1, data[1]=avail1=1, data[2]=value2, data[3]=avail2=0 (unavailable)
        let data = vec![100, 1, 200, 0];
        assert!(decode_results(&labels, &data).is_none());
    }

    #[test]
    fn test_decode_available() {
        let labels = vec![("Scope".to_string(), 0, 1)];
        // data[0]=100 (value), data[1]=1 (available), data[2]=200 (value), data[3]=1 (available)
        let data = vec![100, 1, 200, 1];
        let result = decode_results(&labels, &data).unwrap();
        assert_eq!(result[0].0, "Scope");
        assert_eq!(result[0].1, 100);
        assert_eq!(result[0].2, 200);
    }
}
