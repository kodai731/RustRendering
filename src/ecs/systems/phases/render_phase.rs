use crate::app::App;
use thyllore_vulkan_core::FrameRenderContext;

pub fn build_frame_render_context(app: &App, image_index: usize) -> FrameRenderContext<'_> {
    FrameRenderContext {
        device: &app.rrdevice,
        graphics: &app.data.graphics_resources,
        buffers: &app.data.buffer_registry,
        pipelines: &app.data.pipeline_storage,
        image_index,
    }
}
