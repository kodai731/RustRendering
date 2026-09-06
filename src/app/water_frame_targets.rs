use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::app::App;
use crate::vulkanr::resource::TransientHandle;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct WaterBindingKey {
    tlas: vk::AccelerationStructureKHR,
    hit_table: vk::Buffer,
    history_views: [vk::ImageView; 2],
    scene_color_generation: u64,
    trace_generation: u64,
}

#[derive(Debug, Default)]
pub struct WaterFrameTargets {
    pub scene_color: Option<TransientHandle>,
    pub trace: Option<TransientHandle>,
    bound: Vec<Option<WaterBindingKey>>,
}

impl WaterFrameTargets {
    pub fn forget_bindings(&mut self) {
        self.bound.clear();
    }

    fn clear_handles(&mut self) {
        self.scene_color = None;
        self.trace = None;
    }

    fn is_bound(&self, frame_slot: usize, key: WaterBindingKey) -> bool {
        self.bound.get(frame_slot).copied().flatten() == Some(key)
    }

    fn mark_bound(&mut self, frame_slot: usize, key: WaterBindingKey) {
        if self.bound.len() <= frame_slot {
            self.bound.resize(frame_slot + 1, None);
        }
        self.bound[frame_slot] = Some(key);
    }
}

impl App {
    pub unsafe fn prepare_water_frame_targets(&mut self, frame_slot: usize) -> Result<()> {
        let has_water = !self.data.ecs_world.query_waters().is_empty();
        let scene = self.water_scene_bindings();
        let (Some(water_buffer), Some((tlas, hit_table)), true) =
            (self.data.effect_targets.water.as_ref(), scene, has_water)
        else {
            self.data.water_frame_targets.clear_handles();
            return Ok(());
        };

        let scene_color_desc = water_buffer.scene_color_desc();
        let trace_desc = water_buffer.trace_desc();
        let history_views = water_buffer.history_image_views;
        let history_sampler = water_buffer.history_sampler;

        let transient = &mut self.data.viewport.transient;
        let scene_color = transient.acquire(&self.instance, &self.rrdevice, scene_color_desc)?;
        let trace = transient.acquire(&self.instance, &self.rrdevice, trace_desc)?;
        let scene_color_image = transient.get(scene_color)?;
        let trace_image = transient.get(trace)?;
        self.data.water_frame_targets.scene_color = Some(scene_color);
        self.data.water_frame_targets.trace = Some(trace);

        let key = WaterBindingKey {
            tlas,
            hit_table,
            history_views,
            scene_color_generation: scene_color_image.generation,
            trace_generation: trace_image.generation,
        };
        if self.data.water_frame_targets.is_bound(frame_slot, key) {
            return Ok(());
        }

        let Some(water_ubo) = self.data.raytracing.water_ubo.as_ref() else {
            return Ok(());
        };
        if let Some(descriptor) = self.data.raytracing.water_descriptor.as_ref() {
            descriptor.write_all_at(
                &self.rrdevice,
                frame_slot,
                water_ubo,
                scene_color_image.view,
                history_sampler,
                history_views,
                history_sampler,
                trace_image.view,
                history_sampler,
                tlas,
                hit_table,
            )?;
        }
        if let Some(trace_descriptor) = self.data.raytracing.water_trace_descriptor.as_ref() {
            trace_descriptor.write_all_at(
                &self.rrdevice,
                frame_slot,
                tlas,
                trace_image.view,
                water_ubo,
                hit_table,
            )?;
        }

        self.data.water_frame_targets.mark_bound(frame_slot, key);
        Ok(())
    }

    fn water_scene_bindings(&self) -> Option<(vk::AccelerationStructureKHR, vk::Buffer)> {
        let accel = self.data.raytracing.acceleration_structure.as_ref()?;
        let tlas = accel.tlas.acceleration_structure?;
        let hit_table = accel.hit_shading_table.as_ref()?.buffer;
        Some((tlas, hit_table))
    }
}
