use imgui::Condition;

use crate::ecs::events::{UIEvent, UIEventQueue};
use crate::ecs::resource::gizmo::BoneGizmoData;
use crate::ecs::resource::{
    CoordinateSpace, TransformGizmoMode, TransformGizmoState, WeightHeatmapState,
};
use crate::ecs::World;

use super::viewport_window::ViewportInfo;

const OVERLAY_MARGIN: f32 = 8.0;
const OVERLAY_WIDTH: f32 = 280.0;

pub struct SceneOverlayState {
    pub model_path: String,
    pub load_status: String,
    #[cfg(feature = "auto-rig")]
    pub open_text_to_mesh_dialog: bool,
    #[cfg(feature = "auto-rig")]
    pub open_text_to_animation_dialog: bool,
}

#[cfg(feature = "auto-rig")]
use crate::ecs::resource::{AutoRigState, AutoRigStatus};

pub fn build_scene_overlay(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    overlay_state: &mut SceneOverlayState,
    ecs_world: &World,
    viewport_info: &ViewportInfo,
) {
    let pos_x = viewport_info.position[0] + OVERLAY_MARGIN;
    let pos_y = viewport_info.position[1] + OVERLAY_MARGIN;

    ui.window("Scene Overlay")
        .position([pos_x, pos_y], Condition::Always)
        .size_constraints([OVERLAY_WIDTH, 0.0], [OVERLAY_WIDTH, f32::MAX])
        .no_decoration()
        .bg_alpha(0.7)
        .no_nav()
        .focus_on_appearing(false)
        .save_settings(false)
        .build(|| {
            build_model_section(ui, ui_events, overlay_state, ecs_world);
            ui.separator();

            build_screenshot_section(ui, ui_events);
            ui.separator();

            build_overlay_section(ui, ui_events, ecs_world);

            build_transform_gizmo_section(ui, ui_events, ecs_world);

            build_dof_section(ui, ui_events, ecs_world);

            build_auto_exposure_section(ui, ui_events, ecs_world);

            build_onion_skinning_section(ui, ui_events, ecs_world);
        });
}

fn build_model_section(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    state: &mut SceneOverlayState,
    _ecs_world: &World,
) {
    if ui.button("Open FBX") {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("FBX Files", &["fbx"])
            .pick_file()
        {
            let path_str = path.to_string_lossy().to_string();
            log!("Selected FBX file: {}", path_str);
            ui_events.send(UIEvent::LoadModel { path: path_str });
        }
    }

    ui.same_line();

    if ui.button("Open glTF") {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("glTF Files", &["gltf", "glb"])
            .pick_file()
        {
            let path_str = path.to_string_lossy().to_string();
            log!("Selected glTF file: {}", path_str);
            ui_events.send(UIEvent::LoadModel { path: path_str });
        }
    }

    ui.same_line();

    if ui.button("Open USD") {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("USD Files", &["usdc", "usda", "usd", "usdz"])
            .pick_file()
        {
            let path_str = path.to_string_lossy().to_string();
            log!("Selected USD file: {}", path_str);
            ui_events.send(UIEvent::LoadModel { path: path_str });
        }
    }

    if ui.button("Add GLB") {
        if let Some(paths) = rfd::FileDialog::new()
            .add_filter("GLB Files", &["glb"])
            .pick_files()
        {
            for path in paths {
                let path_str = path.to_string_lossy().to_string();
                log!("Adding GLB file: {}", path_str);
                ui_events.send(UIEvent::LoadModelAdditive { path: path_str });
            }
        }
    }

    #[cfg(feature = "auto-rig")]
    if ui.button("Generate Mesh") {
        state.open_text_to_mesh_dialog = true;
    }

    #[cfg(feature = "auto-rig")]
    {
        ui.same_line();
        if ui.button("Generate Animation") {
            state.open_text_to_animation_dialog = true;
        }
    }

    #[cfg(feature = "auto-rig")]
    build_auto_rig_section(ui, ui_events, _ecs_world);

    let model_name = if state.model_path.is_empty() {
        "None"
    } else {
        &state.model_path
    };
    ui.text_wrapped(format!("Model: {}", model_name));
    ui.text(format!("Status: {}", state.load_status));
}

#[cfg(feature = "auto-rig")]
fn build_auto_rig_section(ui: &imgui::Ui, ui_events: &mut UIEventQueue, ecs_world: &World) {
    use crate::ecs::component::GlbSource;
    use crate::ecs::resource::HierarchyState;
    use crate::ecs::world::Parent;

    let auto_rig_state = ecs_world.resource::<AutoRigState>();
    let status = auto_rig_state.status.clone();
    let joint_count = auto_rig_state.joint_count;
    let bone_count = auto_rig_state.bone_count;
    let gen_time = auto_rig_state.generation_time_ms;
    let error_msg = auto_rig_state.error_message.clone();
    drop(auto_rig_state);

    match status {
        AutoRigStatus::Idle => {
            let hierarchy = ecs_world.resource::<HierarchyState>();
            let selected = hierarchy.selected_entity;
            drop(hierarchy);

            let has_glb_source = selected.map_or(false, |entity| {
                if ecs_world.get_component::<GlbSource>(entity).is_some() {
                    return true;
                }
                if let Some(Parent(parent)) = ecs_world.get_component::<Parent>(entity) {
                    return ecs_world.get_component::<GlbSource>(*parent).is_some();
                }
                false
            });

            if has_glb_source && ui.button("Auto Rig") {
                ui_events.send(UIEvent::AutoRigGenerate {
                    num_sample_points: 65536,
                });
            }
        }

        AutoRigStatus::WaitingForServer => {
            ui.text("Rigging: waiting for server...");
            if ui.button("Cancel##rig") {
                ui_events.send(UIEvent::AutoRigDiscard);
            }
        }

        AutoRigStatus::Rigging => {
            ui.text("Rigging: processing...");
            if ui.button("Cancel##rig") {
                ui_events.send(UIEvent::AutoRigDiscard);
            }
        }

        AutoRigStatus::Previewing => {
            if let Some(gen_time) = gen_time {
                ui.text(format!(
                    "Preview: {} joints, {} bones ({:.1}s)",
                    joint_count.unwrap_or(0),
                    bone_count.unwrap_or(0),
                    gen_time / 1000.0
                ));
            }
            if ui.button("Apply Rig") {
                ui_events.send(UIEvent::AutoRigApply);
            }
            ui.same_line();
            if ui.button("Discard##rig") {
                ui_events.send(UIEvent::AutoRigDiscard);
            }
        }

        AutoRigStatus::Error => {
            if let Some(ref msg) = error_msg {
                ui.text_colored([1.0, 0.3, 0.3, 1.0], format!("Rig error: {}", msg));
            }
            if ui.button("Dismiss##rig") {
                ui_events.send(UIEvent::AutoRigDiscard);
            }
        }
    }
}

fn build_screenshot_section(ui: &imgui::Ui, ui_events: &mut UIEventQueue) {
    if ui.button("Screenshot") {
        ui_events.send(UIEvent::TakeScreenshot);
    }
}

fn build_overlay_section(ui: &imgui::Ui, ui_events: &mut UIEventQueue, ecs_world: &World) {
    if ui.collapsing_header("Overlay", imgui::TreeNodeFlags::DEFAULT_OPEN) {
        if let Some(bone_gizmo) = ecs_world.get_resource::<BoneGizmoData>() {
            let mut visible = bone_gizmo.visible;
            if ui.checkbox("Show Bones", &mut visible) {
                ui_events.send(UIEvent::SetBoneGizmoVisible(visible));
            }
        }
        if let Some(heatmap) = ecs_world.get_resource::<WeightHeatmapState>() {
            let mut enabled = heatmap.enabled;
            if ui.checkbox("Show Weight Heatmap (selected bone)", &mut enabled) {
                ui_events.send(UIEvent::SetWeightHeatmapEnabled(enabled));
            }
        }
    }
}

fn build_transform_gizmo_section(ui: &imgui::Ui, ui_events: &mut UIEventQueue, ecs_world: &World) {
    let Some(state) = ecs_world.get_resource::<TransformGizmoState>() else {
        return;
    };
    let mut state_copy = state.clone();
    drop(state);

    if ui.collapsing_header("Transform Gizmo", imgui::TreeNodeFlags::DEFAULT_OPEN) {
        let translate_label = if state_copy.mode == TransformGizmoMode::Translate {
            "[W] Translate *"
        } else {
            "[W] Translate"
        };
        let rotate_label = if state_copy.mode == TransformGizmoMode::Rotate {
            "[E] Rotate *"
        } else {
            "[E] Rotate"
        };
        let scale_label = if state_copy.mode == TransformGizmoMode::Scale {
            "[R] Scale *"
        } else {
            "[R] Scale"
        };

        if ui.button(translate_label) {
            state_copy.mode = TransformGizmoMode::Translate;
        }
        ui.same_line();
        if ui.button(rotate_label) {
            state_copy.mode = TransformGizmoMode::Rotate;
        }
        ui.same_line();
        if ui.button(scale_label) {
            state_copy.mode = TransformGizmoMode::Scale;
        }

        if ui.is_key_pressed(imgui::Key::W) && !ui.io().key_ctrl {
            state_copy.mode = TransformGizmoMode::Translate;
        }
        if ui.is_key_pressed(imgui::Key::E) && !ui.io().key_ctrl {
            state_copy.mode = TransformGizmoMode::Rotate;
        }
        if ui.is_key_pressed(imgui::Key::R) && !ui.io().key_ctrl {
            state_copy.mode = TransformGizmoMode::Scale;
        }

        let space_label = match state_copy.coordinate_space {
            CoordinateSpace::World => "World",
            CoordinateSpace::Local => "Local",
        };
        if ui.button(format!("Space: {}", space_label)) {
            state_copy.coordinate_space = match state_copy.coordinate_space {
                CoordinateSpace::World => CoordinateSpace::Local,
                CoordinateSpace::Local => CoordinateSpace::World,
            };
        }

        ui.same_line();
        ui.checkbox("Snap", &mut state_copy.snap_enabled);

        if state_copy.snap_enabled {
            match state_copy.mode {
                TransformGizmoMode::Translate => {
                    ui.slider_config("Snap Value", 0.01, 10.0)
                        .build(&mut state_copy.translate_snap_value);
                }
                TransformGizmoMode::Rotate => {
                    ui.slider_config("Snap Degrees", 1.0, 90.0)
                        .build(&mut state_copy.rotate_snap_degrees);
                }
                TransformGizmoMode::Scale => {
                    ui.slider_config("Snap Value", 0.01, 1.0)
                        .build(&mut state_copy.scale_snap_value);
                }
            }
        }

        ui.slider_config("Gizmo Scale", 0.01, 0.3)
            .display_format("%.3f")
            .build(&mut state_copy.gizmo_scale);

        ui_events.send(UIEvent::UpdateTransformGizmoState(Box::new(state_copy)));
    }
}

fn build_dof_section(ui: &imgui::Ui, ui_events: &mut UIEventQueue, ecs_world: &World) {
    use crate::ecs::resource::{DepthOfField, PhysicalCameraParameters};

    if ui.collapsing_header("Depth of Field", imgui::TreeNodeFlags::empty()) {
        if let Some(dof) = ecs_world.get_resource::<DepthOfField>() {
            let mut dof_copy = dof.clone();
            drop(dof);

            ui.checkbox("DOF Enabled", &mut dof_copy.enabled);

            ui.slider_config("Focus Distance", 0.1, 100.0)
                .build(&mut dof_copy.focus_distance);

            ui.slider_config("Max Blur Radius", 1.0, 32.0)
                .build(&mut dof_copy.max_blur_radius);

            ui_events.send(UIEvent::UpdateDepthOfField(dof_copy));
        }

        if let Some(params) = ecs_world.get_resource::<PhysicalCameraParameters>() {
            let mut params_copy = params.clone();
            drop(params);

            ui.slider_config("Aperture (f-stops)", 1.0, 22.0)
                .build(&mut params_copy.aperture_f_stops);

            ui.slider_config("Focal Length (mm)", 10.0, 200.0)
                .build(&mut params_copy.focal_length_mm);

            ui_events.send(UIEvent::UpdatePhysicalCamera(params_copy));
        }
    }
}

fn build_auto_exposure_section(ui: &imgui::Ui, ui_events: &mut UIEventQueue, ecs_world: &World) {
    use crate::ecs::resource::{AutoExposure, Exposure};

    if ui.collapsing_header("Auto Exposure", imgui::TreeNodeFlags::empty()) {
        if let Some(ae) = ecs_world.get_resource::<AutoExposure>() {
            let mut ae_copy = ae.clone();
            drop(ae);

            ui.checkbox("Auto Exposure Enabled", &mut ae_copy.enabled);

            ui.slider_config("Min EV", -10.0, 10.0)
                .build(&mut ae_copy.min_ev);

            ui.slider_config("Max EV", 0.0, 30.0)
                .build(&mut ae_copy.max_ev);

            ui.slider_config("Speed Up", 0.1, 10.0)
                .build(&mut ae_copy.adaptation_speed_up);

            ui.slider_config("Speed Down", 0.1, 10.0)
                .build(&mut ae_copy.adaptation_speed_down);

            ui.slider_config("Low Percent", 0.0, 0.5)
                .build(&mut ae_copy.low_percent);

            ui.slider_config("High Percent", 0.5, 1.0)
                .build(&mut ae_copy.high_percent);

            ui_events.send(UIEvent::UpdateAutoExposure(ae_copy));
        }

        if let Some(exposure) = ecs_world.get_resource::<Exposure>() {
            ui.text(format!("Current Exposure: {:.4}", exposure.exposure_value));
            ui.text(format!("Current EV100: {:.2}", exposure.ev100));
        }
    }
}

fn build_onion_skinning_section(ui: &imgui::Ui, ui_events: &mut UIEventQueue, ecs_world: &World) {
    use crate::ecs::resource::OnionSkinningConfig;

    if ui.collapsing_header("Onion Skinning", imgui::TreeNodeFlags::empty()) {
        if let Some(config) = ecs_world.get_resource::<OnionSkinningConfig>() {
            let mut config_copy = config.clone();
            drop(config);

            ui.checkbox("Onion Skin Enabled", &mut config_copy.enabled);

            let mut past = config_copy.past_count as i32;
            if ui.slider_config("Past Frames", 0, 4).build(&mut past) {
                config_copy.past_count = past.max(0) as u32;
            }

            let mut future = config_copy.future_count as i32;
            if ui.slider_config("Future Frames", 0, 4).build(&mut future) {
                config_copy.future_count = future.max(0) as u32;
            }

            ui.slider_config("Frame Step", 0.001, 0.2)
                .display_format("%.3f")
                .build(&mut config_copy.frame_step);

            ui.slider_config("Ghost Opacity", 0.0, 1.0)
                .build(&mut config_copy.opacity);

            ui.color_edit3("Past Color", &mut config_copy.past_color);
            ui.color_edit3("Future Color", &mut config_copy.future_color);

            ui.text(format!(
                "Total ghosts: {}",
                crate::ecs::compute_total_ghost_count(&config_copy)
            ));

            ui_events.send(UIEvent::UpdateOnionSkinning(config_copy));
        }
    }
}
