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

    if std::env::var("THYLLORE_HIDE_UNSKINNED").is_ok() {
        let mut hidden = 0;
        for mesh in app.data.ecs_assets.meshes.values_mut() {
            if mesh.skeleton_id.is_none() {
                mesh.render_to_gbuffer = false;
                hidden += 1;
            }
        }
        eprintln!("Headless: hid {} unskinned meshes", hidden);
    }

    if std::env::var("THYLLORE_HIDE_BONES").is_ok() {
        use crate::ecs::resource::gizmo::BoneGizmoData;
        if app.data.ecs_world.contains_resource::<BoneGizmoData>() {
            app.data.ecs_world.resource_mut::<BoneGizmoData>().visible = false;
        }
    }

    if let Ok(mode) = std::env::var("THYLLORE_DEBUG_VIEW") {
        use crate::ecs::resource::{DebugViewMode, DebugViewState};
        if let Ok(value) = mode.parse::<i32>() {
            app.data
                .ecs_world
                .resource_mut::<DebugViewState>()
                .debug_view_mode = DebugViewMode::from_int(value);
        }
    }

    let mut imgui = create_empty_imgui_context(width, height);

    for _ in 0..2 {
        imgui.new_frame();
        let draw_data = imgui.render();
        render_one_frame(&mut app, draw_data)?;
    }

    if std::env::var("THYLLORE_NO_AUTOFRAME").is_err() {
        frame_camera_on_bones(&mut app);
    }

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

fn frame_camera_on_bones(app: &mut App) {
    use crate::ecs::resource::gizmo::BoneGizmoData;
    use crate::ecs::resource::Camera;

    let transforms = {
        if !app.data.ecs_world.contains_resource::<BoneGizmoData>() {
            return;
        }
        app.data
            .ecs_world
            .resource::<BoneGizmoData>()
            .cached_global_transforms
            .clone()
    };
    if transforms.is_empty() {
        return;
    }

    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for t in &transforms {
        for axis in 0..3 {
            let v = t[3][axis];
            min[axis] = min[axis].min(v);
            max[axis] = max[axis].max(v);
        }
    }

    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let extent =
        ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2) + (max[2] - min[2]).powi(2))
            .sqrt()
            .max(0.5);

    let mut camera = app.data.ecs_world.resource_mut::<Camera>();
    camera.pivot = cgmath::Vector3::new(center[0], center[1], center[2]);
    camera.distance = extent * 1.2;
    camera.pitch = 0.05;
    camera.yaw = 0.0;
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
