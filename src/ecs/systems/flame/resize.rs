use crate::app::App;
use crate::vulkanr::context::RenderTargets;
use crate::vulkanr::descriptor::FlameImageBindings;
use anyhow::Result;

impl App {
    pub(crate) unsafe fn recreate_flame_on_resize(&mut self) -> Result<()> {
        let (Some(ref flame_buffer), Some(ref flame_descriptor)) = (
            &self.data.viewport.flame_buffer,
            &self.data.raytracing.flame_descriptor,
        ) else {
            return Ok(());
        };
        flame_descriptor.update_image_views(
            &self.rrdevice,
            FlameImageBindings {
                history_image_views: flame_buffer.history_image_views,
                flame_sampler: flame_buffer.sampler,
                sdf_image_view: self.data.raytracing.flame_sdf_image_view,
                sdf_sampler: self.data.raytracing.flame_sdf_sampler,
                scene_depth_view: self
                    .resource::<RenderTargets>()
                    .render
                    .gbuffer_depth_image_view,
            },
        )?;

        if let Some(mut state) = self
            .data
            .ecs_world
            .get_resource_mut::<crate::ecs::resource::FlameHistorySnapshotState>()
        {
            state.previous = None;
        }

        Ok(())
    }
}
