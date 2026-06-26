use std::path::Path;

use anyhow::{Context, Result};
use imgui::ConfigFlags;

use crate::app::App;
use crate::vulkanr::vulkan::*;

const WARMUP_FRAMES: usize = 4;

pub unsafe fn render_usd_to_png_gpu(
    usd_path: &str,
    width: u32,
    height: u32,
    out_path: &Path,
) -> Result<()> {
    let mut app = App::create_headless(width, height).context("Failed to create headless App")?;

    app.load_model(usd_path)
        .with_context(|| format!("Failed to load USD: {}", usd_path))?;

    let mut imgui = create_empty_imgui_context(width, height);

    for _ in 0..WARMUP_FRAMES {
        imgui.new_frame();
        let draw_data = imgui.render();
        render_one_frame(&mut app, draw_data)?;
    }

    app.rrdevice.device.device_wait_idle()?;
    let saved = app.save_offscreen_screenshot()?;

    move_screenshot(&saved, out_path)?;
    Ok(())
}

unsafe fn render_one_frame(app: &mut App, draw_data: &imgui::DrawData) -> Result<()> {
    let image_index = app.begin_frame()?;
    app.update(image_index)?;
    app.render(image_index, draw_data)?;
    Ok(())
}

fn create_empty_imgui_context(width: u32, height: u32) -> imgui::Context {
    let mut imgui = imgui::Context::create();
    imgui.set_ini_filename(None);
    imgui.io_mut().config_flags |= ConfigFlags::NO_MOUSE_CURSOR_CHANGE;
    imgui.io_mut().display_size = [width as f32, height as f32];
    imgui.io_mut().delta_time = 1.0 / 60.0;
    imgui.fonts().build_rgba32_texture();
    imgui
}

fn move_screenshot(saved: &str, out_path: &Path) -> Result<()> {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if Path::new(saved) == out_path {
        return Ok(());
    }
    std::fs::rename(saved, out_path).with_context(|| {
        format!(
            "Failed to move screenshot {} -> {}",
            saved,
            out_path.display()
        )
    })?;
    Ok(())
}
