use crate::ecs::events::{UIEvent, UIEventQueue};
use crate::ecs::resource::{DebugViewMode, DebugViewState};
use crate::ecs::resource::{GridMeshData, MouseInput};
use crate::ecs::World;

pub struct DebugWindowState {
    pub debug_view_mode: DebugViewMode,
}

pub fn build_debug_panel_content(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    state: &mut DebugWindowState,
    ecs_world: &World,
) {
    build_camera_debug_panel(ui, ui_events);
    ui.separator();

    build_debug_view_mode_panel(ui, state);
    ui.separator();

    build_debug_panel(ui, ui_events, ecs_world);
    ui.separator();

    build_grid_debug_panel(ui, ui_events, ecs_world);
    ui.separator();

    build_fbx_debug_panel(ui);
    ui.separator();

    #[cfg(feature = "text-to-motion")]
    {
        build_grpc_server_panel(ui, ecs_world);
        ui.separator();
    }

    build_mouse_info(ui, ecs_world);
}

fn build_camera_debug_panel(ui: &imgui::Ui, ui_events: &mut UIEventQueue) {
    ui.text("Camera:");
    if ui.button("Reset Camera") {
        ui_events.send(UIEvent::ResetCamera);
    }
    ui.same_line();
    if ui.button("Reset Up") {
        ui_events.send(UIEvent::ResetCameraUp);
    }
    ui.same_line();
    if ui.button("To Model") {
        ui_events.send(UIEvent::MoveCameraToModel);
    }
}

fn build_debug_view_mode_panel(ui: &imgui::Ui, state: &mut DebugWindowState) {
    ui.text("Debug View Mode:");
    let mut current_mode = state.debug_view_mode.as_int();

    if ui.radio_button("Final (Lit + Shadow)", &mut current_mode, 0) {
        state.debug_view_mode = DebugViewMode::Final;
    }
    if ui.radio_button("Position (World Space)", &mut current_mode, 1) {
        state.debug_view_mode = DebugViewMode::Position;
    }
    if ui.radio_button("Normal (World Space)", &mut current_mode, 2) {
        state.debug_view_mode = DebugViewMode::Normal;
    }
    if ui.radio_button("Shadow Mask", &mut current_mode, 3) {
        state.debug_view_mode = DebugViewMode::ShadowMask;
    }
    if ui.radio_button("N dot L (Green=Lit, Red=Back)", &mut current_mode, 4) {
        state.debug_view_mode = DebugViewMode::NdotL;
    }
    if ui.radio_button("Light Direction", &mut current_mode, 5) {
        state.debug_view_mode = DebugViewMode::LightDirection;
    }
    if ui.radio_button("View Depth (Green=GBuffer depth)", &mut current_mode, 6) {
        state.debug_view_mode = DebugViewMode::ViewDepth;
    }
    if ui.radio_button("ObjectID (Color per ID)", &mut current_mode, 7) {
        state.debug_view_mode = DebugViewMode::ObjectID;
    }
    if ui.radio_button("Selection View (Orange=Selected)", &mut current_mode, 8) {
        state.debug_view_mode = DebugViewMode::SelectionView;
    }
    if ui.radio_button("SelectionUBO (R=count, G=id0)", &mut current_mode, 9) {
        state.debug_view_mode = DebugViewMode::SelectionUBO;
    }
}

fn build_debug_panel(ui: &imgui::Ui, ui_events: &mut UIEventQueue, ecs_world: &World) {
    ui.text("Debug Info:");

    if let Some(mut debug_view) = ecs_world.get_resource_mut::<DebugViewState>() {
        ui.checkbox("Show Click Debug", &mut debug_view.show_click_debug);
    }

    if ui.button("Debug Shadow Info") {
        ui_events.send(crate::ecs::events::UIEvent::DebugShadowInfo);
    }

    ui.same_line();

    if ui.button("Debug Billboard Depth") {
        ui_events.send(crate::ecs::events::UIEvent::DebugBillboardDepth);
    }

    if ui.button("Dump Debug Information") {
        ui_events.send(crate::ecs::events::UIEvent::DumpDebugInfo);
    }

    ui.same_line();

    if ui.button("Dump Animation Debug") {
        ui_events.send(crate::ecs::events::UIEvent::DumpAnimationDebug);
    }

    if ui.button("Add Test Constraints") {
        ui_events.send(crate::ecs::events::UIEvent::CreateTestConstraints);
    }
    ui.same_line();
    if ui.button("Clear Constraints") {
        ui_events.send(crate::ecs::events::UIEvent::ClearTestConstraints);
    }

    if ui.button("Add Spring Bones") {
        ui_events.send(crate::ecs::events::UIEvent::AddTestSpringBones);
    }
    ui.same_line();
    if ui.button("Clear Spring Bones") {
        ui_events.send(crate::ecs::events::UIEvent::ClearSpringBones);
    }

    build_spring_bone_bake_panel(ui, ui_events, ecs_world);
}

fn build_spring_bone_bake_panel(ui: &imgui::Ui, ui_events: &mut UIEventQueue, ecs_world: &World) {
    use crate::ecs::events::UIEvent;
    use crate::ecs::resource::{SpringBoneMode, SpringBoneState};

    let Some(state) = ecs_world.get_resource::<SpringBoneState>() else {
        return;
    };

    ui.separator();
    match state.mode {
        SpringBoneMode::Realtime => {
            ui.text("Spring Bone: Realtime");
            ui.text_colored([0.5, 0.8, 0.5, 1.0], "  Simulating...");
            if ui.button("Bake Spring Bones") {
                ui_events.send(UIEvent::SpringBoneBake);
            }
        }
        SpringBoneMode::Baked => {
            let clip_id = state.baked_clip_source_id.unwrap_or(0);
            ui.text(format!("Spring Bone: Baked (clip_id={})", clip_id));
            ui.text_colored(
                [0.7, 0.7, 0.5, 1.0],
                "  Editing will switch to BakedOverride",
            );
            if ui.button("Discard Bake") {
                ui_events.send(UIEvent::SpringBoneDiscardBake);
            }
            ui.same_line();
            if ui.button("Save Bake (.ron)") {
                ui_events.send(UIEvent::SpringBoneSaveBake);
            }
        }
        SpringBoneMode::BakedOverride => {
            let clip_id = state.baked_clip_source_id.unwrap_or(0);
            ui.text(format!("Spring Bone: BakedOverride (clip_id={})", clip_id));
            ui.text_colored([1.0, 0.7, 0.3, 1.0], "  Manually edited");
            if ui.button("Re-bake") {
                ui_events.send(UIEvent::SpringBoneRebake);
            }
            ui.same_line();
            if ui.button("Discard Bake") {
                ui_events.send(UIEvent::SpringBoneDiscardBake);
            }
            ui.same_line();
            if ui.button("Save Bake (.ron)") {
                ui_events.send(UIEvent::SpringBoneSaveBake);
            }
        }
    }
}

fn build_grid_debug_panel(ui: &imgui::Ui, ui_events: &mut UIEventQueue, ecs_world: &World) {
    use crate::ecs::events::UIEvent;

    ui.text("Grid:");
    if let Some(grid) = ecs_world.get_resource::<GridMeshData>() {
        let mut show_y = grid.show_y_axis_grid;
        if ui.checkbox("Show Y-Axis Grid", &mut show_y) {
            ui_events.send(UIEvent::SetGridShowYAxis(show_y));
        }
    }
}

fn build_fbx_debug_panel(ui: &imgui::Ui) {
    use crate::debugview::FBX_DEBUG;

    ui.text("FBX Debug Logs:");

    let mut fbx_anim = FBX_DEBUG.animation_enabled();
    let mut fbx_hier = FBX_DEBUG.hierarchy_enabled();
    let mut fbx_skin = FBX_DEBUG.skinning_enabled();
    let mut fbx_trans = FBX_DEBUG.transform_enabled();

    if ui.checkbox("Animation", &mut fbx_anim) {
        FBX_DEBUG.set_animation(fbx_anim);
    }
    if ui.checkbox("Hierarchy", &mut fbx_hier) {
        FBX_DEBUG.set_hierarchy(fbx_hier);
    }
    if ui.checkbox("Skinning", &mut fbx_skin) {
        FBX_DEBUG.set_skinning(fbx_skin);
    }
    if ui.checkbox("Transform", &mut fbx_trans) {
        FBX_DEBUG.set_transform(fbx_trans);
    }
}

#[cfg(feature = "text-to-motion")]
fn build_grpc_server_panel(ui: &imgui::Ui, ecs_world: &World) {
    use crate::ecs::resource::GrpcServerProcess;

    ui.text("gRPC Server:");

    let Some(mut server) = ecs_world.get_resource_mut::<GrpcServerProcess>() else {
        return;
    };

    let running = server.is_running();

    if running {
        ui.text_colored([0.5, 1.0, 0.5, 1.0], "Running");
        if ui.button("Stop Server") {
            server.stop();
        }
    } else {
        ui.text_colored([0.7, 0.7, 0.7, 1.0], "Stopped");
        if let Some(err) = &server.last_error {
            ui.text_colored([1.0, 0.3, 0.3, 1.0], err);
        }
        if ui.button("Start Server") {
            let working_dir = find_training_repo_path();
            match server.start(&working_dir, "configs/server.yaml") {
                Ok(()) => {}
                Err(e) => {
                    log_error!("Failed to start gRPC server: {}", e);
                    server.last_error = Some(e);
                }
            }
        }
    }
}

#[cfg(feature = "text-to-motion")]
fn find_training_repo_path() -> String {
    if let Ok(path) = std::env::var("ANIMATION_TRAINING_PATH") {
        return path;
    }

    let output = std::process::Command::new("wsl")
        .args([
            "--",
            "bash",
            "-l",
            "-c",
            "echo $HOME/Projects/AnimationModelTraining",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output();

    if let Ok(out) = output {
        let wsl_path = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !wsl_path.is_empty() {
            return wsl_path;
        }
    }

    "../AnimationModelTraining".to_string()
}

fn build_mouse_info(ui: &imgui::Ui, ecs_world: &World) {
    let mouse = ecs_world.resource::<MouseInput>();
    ui.text(format!(
        "Mouse Position: ({:.1},{:.1})",
        mouse.position[0], mouse.position[1]
    ));
    ui.text(format!("is left clicked: ({:.1})", mouse.left_pressed));
    ui.text(format!("is wheel clicked: ({:.1})", mouse.middle_pressed));
}

pub fn build_click_debug_overlay(ui: &imgui::Ui, ecs_world: &World) {
    let show = ecs_world
        .get_resource::<DebugViewState>()
        .map(|s| s.show_click_debug)
        .unwrap_or(false);
    if !show {
        return;
    }

    use std::sync::atomic::{AtomicBool, Ordering};
    static IMGUI_SIZE_LOGGED: AtomicBool = AtomicBool::new(false);
    if !IMGUI_SIZE_LOGGED.load(Ordering::Relaxed) {
        let display_size = ui.io().display_size;
        log!(
            "ImGui display size: {:.1} x {:.1}",
            display_size[0],
            display_size[1]
        );
        IMGUI_SIZE_LOGGED.store(true, Ordering::Relaxed);
    }

    let rect = ecs_world
        .get_resource::<DebugViewState>()
        .and_then(|s| s.billboard_click_rect);
    if let Some(rect) = rect {
        let draw_list = ui.get_foreground_draw_list();
        draw_list
            .add_rect([rect[0], rect[1]], [rect[2], rect[3]], [1.0, 0.0, 0.0, 0.8])
            .filled(true)
            .build();
        draw_list
            .add_rect([rect[0], rect[1]], [rect[2], rect[3]], [1.0, 1.0, 0.0, 1.0])
            .thickness(2.0)
            .build();
    }
}
