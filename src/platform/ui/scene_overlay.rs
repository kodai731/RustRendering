use imgui::Condition;

use crate::ecs::component::FlameParam;
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
    pub flame_preset_index: usize,
    pub texture_fit_path: String,
    pub texture_fit_blend: f32,
    pub texture_fit_groups: [bool; 4],
    pub texture_fit_profile: bool,
    pub texture_fit_scan: Vec<String>,
    pub texture_fit_scan_done: bool,
    pub texture_fit_browser_open: bool,
    pub texture_fit_browser_dir: String,
    pub texture_fit_browser_selected: String,
    pub texture_fit_browser_show_all: bool,
    pub texture_fit_browser_show_hidden: bool,
    pub texture_fit_path_validated: String,
    pub texture_fit_path_info: String,
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
        .always_auto_resize(true)
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

            build_flame_section(ui, ui_events, overlay_state, ecs_world, viewport_info);
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

        let gizmo_hotkeys_enabled =
            !ui.io().key_ctrl && !ui.is_mouse_down(imgui::MouseButton::Right);
        if ui.is_key_pressed(imgui::Key::W) && gizmo_hotkeys_enabled {
            state_copy.mode = TransformGizmoMode::Translate;
        }
        if ui.is_key_pressed(imgui::Key::E) && gizmo_hotkeys_enabled {
            state_copy.mode = TransformGizmoMode::Rotate;
        }
        if ui.is_key_pressed(imgui::Key::R) && gizmo_hotkeys_enabled {
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

fn build_flame_section(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    overlay_state: &mut SceneOverlayState,
    ecs_world: &World,
    viewport_info: &ViewportInfo,
) {
    use crate::ecs::component::FlameEffect;
    use crate::ecs::resource::{FlameRenderSettings, FlameShadingMode};

    if ui.collapsing_header("Flame", imgui::TreeNodeFlags::empty()) {
        if let Some(settings) = ecs_world.get_resource::<FlameRenderSettings>() {
            let mut settings_copy = *settings;
            drop(settings);

            if let Some(_token) = ui.begin_combo("Shading Mode", settings_copy.shading_mode.label())
            {
                for mode in FlameShadingMode::ALL {
                    let selected = mode == settings_copy.shading_mode;
                    if ui
                        .selectable_config(mode.label())
                        .selected(selected)
                        .build()
                    {
                        settings_copy.shading_mode = mode;
                    }
                }
            }

            {
                use crate::ecs::resource::FlameDebugView;
                if let Some(_token) = ui.begin_combo("Debug View", settings_copy.debug_view.label())
                {
                    for view in FlameDebugView::ALL {
                        let selected = view == settings_copy.debug_view;
                        if ui
                            .selectable_config(view.label())
                            .selected(selected)
                            .build()
                        {
                            settings_copy.debug_view = view;
                        }
                    }
                }
            }

            {
                use thyllore_effect_core::flame_wave::{
                    read_env_wave_jitter, read_env_wave_jitter_freq, set_wave_jitter,
                    set_wave_jitter_freq,
                };
                let mut jitter = read_env_wave_jitter();
                if ui
                    .slider_config("Jitter Depth", 0.0f32, 2.0f32)
                    .build(&mut jitter)
                {
                    set_wave_jitter(jitter);
                }
                let mut jitter_freq = read_env_wave_jitter_freq();
                if ui
                    .slider_config("Jitter Freq", 0.25f32, 6.0f32)
                    .build(&mut jitter_freq)
                {
                    set_wave_jitter_freq(jitter_freq);
                }
            }

            match settings_copy.shading_mode {
                FlameShadingMode::ReferenceRaymarch => {
                    let mut steps = settings_copy.reference_step_count as i32;
                    ui.slider_config("Reference Steps", 8, 512)
                        .build(&mut steps);
                    settings_copy.reference_step_count = steps.max(1) as u32;
                }
                FlameShadingMode::NoiseRaymarch => {
                    let mut steps = settings_copy.noise_step_count as i32;
                    ui.slider_config("Noise Steps", 4, 64).build(&mut steps);
                    settings_copy.noise_step_count = steps.max(1) as u32;
                }
                FlameShadingMode::Analytic
                | FlameShadingMode::DebugThickness
                | FlameShadingMode::DebugDepthClamp => {}
            }

            ui_events.send(UIEvent::UpdateFlameRenderSettings(settings_copy));
        }

        let flames = ecs_world.query_flames();
        let selected_flame_entity = crate::ecs::systems::resolve_selected_flame(ecs_world);
        let clamped_index = selected_flame_entity
            .and_then(|entity| flames.iter().position(|&e| e == entity))
            .unwrap_or(0);

        // Instance selector combo (when >1 flame)
        if flames.len() > 1 {
            let mut current = clamped_index;
            let items: Vec<String> = flames
                .iter()
                .enumerate()
                .map(|(i, &entity)| {
                    ecs_world
                        .get_component::<crate::ecs::world::Name>(entity)
                        .map(|n| n.0.clone())
                        .unwrap_or_else(|| format!("Flame {}", i + 1))
                })
                .collect();
            if ui.combo_simple_string("Instance", &mut current, &items) {
                ui_events.send(UIEvent::SelectFlameInstance(current));
            }
        }

        // Flame Preset selector
        let presets: Vec<String> = thyllore_effect_core::FLAME_PRESET_NAMES
            .iter()
            .map(|s| s.to_string())
            .collect();
        // The slider block below re-sends the (pre-apply) effect every frame;
        // that send must be skipped on the frame an Apply button fires or it
        // overwrites the applied preset/fit in the same dispatch.
        let mut effect_applied_this_frame = false;
        {
            let mut preset_index = overlay_state.flame_preset_index;
            let preset_changed =
                ui.combo_simple_string("Flame Preset", &mut preset_index, &presets);
            overlay_state.flame_preset_index = preset_index;
            if preset_changed {
                if selected_flame_entity.is_some() {
                    // Keyed scalar curves re-stamp their channels every
                    // frame and would silently pin the old look, so a
                    // preset stamp also clears them (undo restores).
                    ui_events.send(UIEvent::ClearScalarKeys);
                    ui_events.send(UIEvent::ApplyFlamePreset(presets[preset_index].clone()));
                    effect_applied_this_frame = true;
                }
            }

            ui.separator();
            ui.text("Texture Fit");

            // Scan textures on first frame
            if !overlay_state.texture_fit_scan_done {
                let scan_dir = std::path::Path::new(crate::paths::FLAMES_TEXTURE_DIR);
                if let Ok(entries) = std::fs::read_dir(scan_dir) {
                    for entry in entries.flatten() {
                        if let Some(ext) = entry.path().extension() {
                            if ext == "png" {
                                if let Some(name) = entry.file_name().to_str() {
                                    overlay_state.texture_fit_scan.push(name.to_string());
                                }
                            }
                        }
                    }
                }
                overlay_state.texture_fit_scan_done = true;
            }

            // Combo box for texture selection
            let mut scan_items: Vec<String> = vec!["(custom path)".to_string()];
            scan_items.extend(overlay_state.texture_fit_scan.iter().cloned());
            let mut scan_selected = 0usize;
            if ui.combo_simple_string("Fit Texture", &mut scan_selected, &scan_items) {
                if scan_selected > 0 {
                    let name = &overlay_state.texture_fit_scan[scan_selected - 1];
                    overlay_state.texture_fit_path =
                        format!("{}/{}", crate::paths::FLAMES_TEXTURE_DIR, name);
                } else {
                    overlay_state.texture_fit_path.clear();
                }
            }
            ui.same_line();
            if ui.small_button("Rescan") {
                overlay_state.texture_fit_scan_done = false;
                overlay_state.texture_fit_scan.clear();
            }

            ui.input_text("Fit Image (png)", &mut overlay_state.texture_fit_path)
                .build();
            if ui.small_button("Browse...") {
                overlay_state.texture_fit_browser_open = true;
                if overlay_state.texture_fit_browser_dir.is_empty() {
                    overlay_state.texture_fit_browser_dir =
                        crate::paths::FLAMES_TEXTURE_DIR.to_string();
                }
                overlay_state.texture_fit_browser_dir =
                    canonical_dir_or(&overlay_state.texture_fit_browser_dir);
            }
            ui.same_line();

            // Validation indicator: existence plus a lightweight PNG header read,
            // cached per path so the header is only parsed when the path changes.
            if overlay_state.texture_fit_path != overlay_state.texture_fit_path_validated {
                overlay_state.texture_fit_path_info =
                    validate_texture_fit_path(&overlay_state.texture_fit_path);
                overlay_state.texture_fit_path_validated = overlay_state.texture_fit_path.clone();
            }
            if overlay_state.texture_fit_path.is_empty() {
                ui.text_disabled("enter a texture path");
            } else if overlay_state.texture_fit_path_info.starts_with("ok:") {
                ui.text_colored([0.3, 0.9, 0.3, 1.0], &overlay_state.texture_fit_path_info);
            } else {
                ui.text_colored([0.9, 0.3, 0.3, 1.0], &overlay_state.texture_fit_path_info);
            }

            build_texture_fit_browser(ui, overlay_state);

            ui.slider("Fit Blend", 0.0, 1.0, &mut overlay_state.texture_fit_blend);
            ui.checkbox("Silhouette", &mut overlay_state.texture_fit_groups[0]);
            ui.checkbox("Color", &mut overlay_state.texture_fit_groups[1]);
            {
                let _disabled = ui.begin_disabled(overlay_state.texture_fit_profile);
                ui.checkbox("Turbulence", &mut overlay_state.texture_fit_groups[2]);
            }
            if overlay_state.texture_fit_profile && ui.is_item_hovered() {
                ui.tooltip_text(
                    "Ignored in profile (reproduction) mode: the turbulence \
                     estimate is far below the calibrated pattern amplitude \
                     and would crush the noise",
                );
            }
            ui.checkbox("Tilt", &mut overlay_state.texture_fit_groups[3]);

            // Fidelity radio button
            let mut fidelity_mode: i32 = if overlay_state.texture_fit_profile {
                1
            } else {
                0
            };
            if ui.radio_button("statistics (projection)", &mut fidelity_mode, 0) {
                overlay_state.texture_fit_profile = false;
            }
            ui.same_line();
            if ui.radio_button("profile (reproduction)", &mut fidelity_mode, 1) {
                overlay_state.texture_fit_profile = true;
            }

            if ui.button("Apply Texture Fit") {
                let path = overlay_state.texture_fit_path.clone();
                let blend = overlay_state.texture_fit_blend;
                let groups = thyllore_effect_core::TextureFitGroups {
                    silhouette: overlay_state.texture_fit_groups[0],
                    color: overlay_state.texture_fit_groups[1],
                    turbulence: overlay_state.texture_fit_groups[2],
                    tilt: overlay_state.texture_fit_groups[3],
                };
                if selected_flame_entity.is_some() {
                    ui_events.send(UIEvent::ApplyFlameTextureFit {
                        path: path.clone(),
                        blend,
                        groups: [
                            groups.silhouette,
                            groups.color,
                            groups.turbulence,
                            groups.tilt,
                        ],
                        profile: overlay_state.texture_fit_profile,
                    });
                    effect_applied_this_frame = true;
                }
            }

            if let Some(selected_flame) = selected_flame_entity {
                if let Some(effect) = ecs_world.get_component::<FlameEffect>(selected_flame) {
                    let mut effect_copy = effect.clone();

                    let mut position = [
                        effect_copy.position.x,
                        effect_copy.position.y,
                        effect_copy.position.z,
                    ];
                    if ui.input_float3("Position", &mut position).build() {
                        effect_copy.position =
                            cgmath::Vector3::new(position[0], position[1], position[2]);
                    }

                    let emitter_labels: [&str; 3] = ["Cylinder", "Ring", "Mesh SDF"];
                    let mut emitter_selected = effect_copy.emitter_kind as usize;
                    if ui.combo_simple_string("Emitter", &mut emitter_selected, &emitter_labels) {
                        effect_copy.emitter_kind = emitter_selected as u32;
                    }

                    if effect_copy.emitter_kind == 1 {
                        ui.slider_config("Ring Radius", 0.2, 5.0)
                            .display_format("%.2f")
                            .build(&mut effect_copy.ring_major_radius);
                        ui.same_line();
                        let mut ring_speed = effect_copy.ring_angular_speed;
                        ui.slider_config("Ring Speed", 0.0, 6.28)
                            .display_format("%.2f")
                            .build(&mut ring_speed);
                        effect_copy.ring_angular_speed = ring_speed;
                    }

                    ui.slider_config("Height", 0.05, 10.0)
                        .display_format("%.2f")
                        .build(&mut effect_copy.height);
                    ui.same_line();
                    if ui.small_button("K##Height") {
                        ui_events.send(UIEvent::InsertScalarKey {
                            property_type: FlameParam::Height.property_type(),
                            value: effect_copy.height,
                        });
                    }

                    ui.slider_config("Radius", 0.05, 10.0)
                        .display_format("%.2f")
                        .build(&mut effect_copy.radius);
                    ui.same_line();
                    if ui.small_button("K##Radius") {
                        ui_events.send(UIEvent::InsertScalarKey {
                            property_type: FlameParam::Radius.property_type(),
                            value: effect_copy.radius,
                        });
                    }

                    ui.slider_config("Intensity", 0.0, 10.0)
                        .build(&mut effect_copy.intensity);
                    ui.same_line();
                    if ui.small_button("K##Intensity") {
                        ui_events.send(UIEvent::InsertScalarKey {
                            property_type: FlameParam::Intensity.property_type(),
                            value: effect_copy.intensity,
                        });
                    }

                    let mut color_changed = false;
                    color_changed |= ui.color_edit3("Base Color", &mut effect_copy.color_base);
                    color_changed |= ui.color_edit3("Tip Color", &mut effect_copy.color_tip);
                    if color_changed {
                        effect_copy.use_blackbody = false;
                    }

                    ui.slider_config("Noise Amplitude", 0.0, 3.0)
                        .build(&mut effect_copy.noise_amplitude);
                    ui.same_line();
                    if ui.small_button("K##NoiseAmplitude") {
                        ui_events.send(UIEvent::InsertScalarKey {
                            property_type: FlameParam::NoiseAmplitude.property_type(),
                            value: effect_copy.noise_amplitude,
                        });
                    }

                    ui.slider_config("Noise Contrast", 0.25, 4.0)
                        .display_format("%.2f")
                        .build(&mut effect_copy.noise_contrast);

                    ui.slider_config("Swirl", 0.0, 1.5)
                        .display_format("%.2f")
                        .build(&mut effect_copy.swirl_gain);
                    if ui.is_item_hovered() {
                        ui.tooltip_text(
                            "Medium swirl share: strain budget spent on azimuthal shear \
                             (0 = off; raising it thins the carve warp)",
                        );
                    }

                    ui.slider_config("Noise Shaping", 0.0, 6.0)
                        .display_format("%.2f")
                        .build(&mut effect_copy.noise_shaping_scale);
                    if ui.is_item_hovered() {
                        ui.tooltip_text(
                            "tanh shaping scale of the wave noise (0 = built-in 0.6). \
                             The default clips the pattern at +-0.28 and crushes interior \
                             contrast; ~3 lets the pattern breathe (measured +58%)",
                        );
                    }

                    let mut vortex = (effect_copy.twist_gain
                        / thyllore_effect_core::VORTEX_MACRO_MAX_GAIN)
                        .clamp(0.0, 1.0);
                    if ui
                        .slider_config("Vortex", 0.0, 1.0)
                        .display_format("%.2f")
                        .build(&mut vortex)
                    {
                        let (gain, speed) = thyllore_effect_core::vortex_macro_levers(vortex);
                        effect_copy.twist_gain = gain;
                        effect_copy.twist_speed = speed;
                    }
                    if ui.is_item_hovered() {
                        ui.tooltip_text(
                            "Vortex macro: one knob writing both twist levers along a \
                             faster-and-deeper curve (stateless; the fine sliders below \
                             stay the source of truth)",
                        );
                    }

                    ui.slider_config("Twist", 0.0, 8.0)
                        .display_format("%.2f")
                        .build(&mut effect_copy.twist_gain);
                    if ui.is_item_hovered() {
                        ui.tooltip_text(
                            "Azimuthal twist of the noise pattern around the axis \
                             (radians at the tip; a rotation never folds, so any \
                             amplitude is structurally safe; 0 = off)",
                        );
                    }

                    ui.slider_config("Twist Speed", 0.0, 4.0)
                        .display_format("%.2f")
                        .build(&mut effect_copy.twist_speed);
                    if ui.is_item_hovered() {
                        ui.tooltip_text(
                            "Twist rotation rate scale (0 = follow Swirl Speed; > 0 gives \
                             the twist its own rate so depth and speed tune independently)",
                        );
                    }

                    ui.slider_config("Meander", 0.0, 2.0)
                        .display_format("%.2f")
                        .build(&mut effect_copy.meander_amp);
                    if ui.is_item_hovered() {
                        ui.tooltip_text("Horizontal meandering motion of the flame (0 = off)");
                    }

                    ui.slider_config("Swirl Speed", 0.0, 4.0)
                        .display_format("%.2f")
                        .build(&mut effect_copy.swirl_speed);
                    if ui.is_item_hovered() {
                        ui.tooltip_text(
                            "How fast the swirl layers counter-rotate against the rise \
                             (time-only: costs no strain budget)",
                        );
                    }
                    ui.slider_config("Spread", 0.0, 3.0)
                        .display_format("%.2f")
                        .build(&mut effect_copy.spread_gain);
                    if ui.is_item_hovered() {
                        ui.tooltip_text(
                            "Medium spread toward the tip: noise features enlarge, drift \
                             outward and dissolve as they rise (0 = rigid scroll)",
                        );
                    }

                    ui.slider_config("Support", 1.0, 2.5)
                        .display_format("%.2f")
                        .build(&mut effect_copy.support_margin);
                    if ui.is_item_hovered() {
                        ui.tooltip_text(
                            "Flame density support radius: multiplier for the biweight support \
                             radius (how much extra space is allowed for carving). 1.0 is \
                             default; higher values result in larger support and may leave \
                             chunks at the outer edges.",
                        );
                    }

                    ui.slider_config("Time Scale", 0.0, 4.0)
                        .build(&mut effect_copy.time_scale);

                    if ui.button("Clear Flame Keys") {
                        ui_events.send(UIEvent::ClearScalarKeys);
                    }
                    ui.same_line();
                    if ui.button("Random Keys (Debug)") {
                        let seed = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_millis() as u64)
                            .unwrap_or(0);
                        ui_events.send(UIEvent::InsertScalarDebugKeys { seed });
                    }
                    if ui.button("Curves") {
                        ui_events.send(UIEvent::OpenScalarCurveEditor);
                    }
                    if ui.button("Add Flame") {
                        ui_events.send(UIEvent::AddFlame);
                    }
                    ui.same_line();
                    if ui.button("Dump Probe") {
                        ui_events.send(UIEvent::DumpFlameWallProbe {
                            viewport_size: viewport_info.size,
                        });
                    }
                    if ui.is_item_hovered() {
                        ui.tooltip_text(
                            "Dump camera pose + wall-regime ray diagnostics to log/flame/",
                        );
                    }

                    // Trail checkbox and slider
                    let trail_state = ecs_world
                        .get_component::<crate::ecs::component::flame_trail::FlameTrail>(
                            selected_flame,
                        )
                        .map(|t| (t.state.enabled, t.state.fade_seconds))
                        .unwrap_or((false, 0.8));
                    let mut trail_enabled = trail_state.0;
                    let mut trail_fade = trail_state.1;
                    if ui.checkbox("Trail", &mut trail_enabled) {
                        ui_events.send(UIEvent::UpdateFlameTrailEnabled(trail_enabled));
                    }
                    ui.slider_config("Trail Fade", 0.1, 5.0)
                        .build(&mut trail_fade);
                    if (trail_fade - trail_state.1).abs() > 0.01 {
                        ui_events.send(UIEvent::UpdateFlameTrailFade(trail_fade));
                    }

                    // GPU Timings section (read-only)
                    let timings = ecs_world.get_resource::<crate::ecs::resource::GpuPassTimings>();
                    if let Some(timings) = timings {
                        if !timings.passes.is_empty() {
                            ui.separator();
                            ui.text("GPU Timings");
                            for (label, ms) in &timings.passes {
                                ui.text(format!("  {} {:.3} ms", label, ms));
                            }
                        }
                    }

                    if !effect_applied_this_frame {
                        ui_events.send(UIEvent::UpdateFlameEffect(Box::new(effect_copy)));
                    }
                }
            }
        }
    }
}

/// Canonicalized directory, falling back to the typed text when the path
/// cannot be resolved (broken symlink, permissions, not yet existing).
fn canonical_dir_or(dir: &str) -> String {
    std::fs::canonicalize(dir)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| dir.to_string())
}

/// Lightweight validation for the fit path indicator: existence plus a PNG
/// header read (no pixel decode). "ok: ..." prefixed on success.
fn validate_texture_fit_path(path: &str) -> String {
    if path.is_empty() {
        return String::new();
    }
    if !std::path::Path::new(path).is_file() {
        return String::from("file not found");
    }
    match std::fs::File::open(path)
        .map_err(|e| e.to_string())
        .and_then(|file| {
            png::Decoder::new(file)
                .read_info()
                .map_err(|e| e.to_string())
        }) {
        Ok(reader) => {
            let info = reader.info();
            format!("ok: {}x{} {:?}", info.width, info.height, info.color_type)
        }
        Err(error) => format!("not a readable png: {error}"),
    }
}

const TEXTURE_FIT_BROWSER_MAX_ENTRIES: usize = 2000;

/// In-app file browser for the fit texture (G9): breadcrumb + direct path
/// input, png filter, directory-first listing, double-click to descend /
/// confirm. Selection only fills the path field — applying stays on the
/// explicit Apply button. Unreadable entries render disabled instead of
/// failing the listing.
fn build_texture_fit_browser(ui: &imgui::Ui, overlay_state: &mut SceneOverlayState) {
    if !overlay_state.texture_fit_browser_open {
        return;
    }
    let mut open = true;
    let mut confirmed: Option<String> = None;
    ui.window("Select Fit Texture")
        .size([560.0, 430.0], imgui::Condition::FirstUseEver)
        .opened(&mut open)
        .build(|| {
            let dir_now = overlay_state.texture_fit_browser_dir.clone();
            let mut jump: Option<String> = None;

            if ui.small_button("/") {
                jump = Some(String::from("/"));
            }
            let mut accumulated = String::new();
            for (index, part) in dir_now.split('/').filter(|p| !p.is_empty()).enumerate() {
                accumulated.push('/');
                accumulated.push_str(part);
                ui.same_line();
                if ui.small_button(format!("{part}##crumb{index}")) {
                    jump = Some(accumulated.clone());
                }
            }

            ui.input_text(
                "##fit_browser_dir",
                &mut overlay_state.texture_fit_browser_dir,
            )
            .build();
            ui.same_line();
            if ui.small_button("Go") {
                jump = Some(overlay_state.texture_fit_browser_dir.clone());
            }
            ui.checkbox("all files", &mut overlay_state.texture_fit_browser_show_all);
            ui.same_line();
            ui.checkbox("hidden", &mut overlay_state.texture_fit_browser_show_hidden);
            ui.same_line();
            if ui.small_button("Up") {
                if let Some(parent) = std::path::Path::new(&dir_now).parent() {
                    jump = Some(parent.display().to_string());
                }
            }

            ui.child_window("##fit_browser_list")
                .size([0.0, -34.0])
                .build(|| {
                    let read = match std::fs::read_dir(&dir_now) {
                        Ok(read) => read,
                        Err(error) => {
                            ui.text_colored(
                                [0.9, 0.3, 0.3, 1.0],
                                format!("cannot read directory: {error}"),
                            );
                            return;
                        }
                    };
                    let mut rows: Vec<(String, bool, Option<u64>)> = Vec::new();
                    let mut truncated = false;
                    for entry in read.flatten() {
                        let name = match entry.file_name().into_string() {
                            Ok(name) => name,
                            Err(_) => continue,
                        };
                        if !overlay_state.texture_fit_browser_show_hidden && name.starts_with('.') {
                            continue;
                        }
                        let metadata = entry.metadata().ok();
                        let is_dir = metadata.as_ref().is_some_and(|m| m.is_dir());
                        if !is_dir
                            && !overlay_state.texture_fit_browser_show_all
                            && !name.to_ascii_lowercase().ends_with(".png")
                        {
                            continue;
                        }
                        if rows.len() >= TEXTURE_FIT_BROWSER_MAX_ENTRIES {
                            truncated = true;
                            break;
                        }
                        rows.push((name, is_dir, metadata.map(|m| m.len())));
                    }
                    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                    for (name, is_dir, size) in &rows {
                        let label = if *is_dir {
                            format!("{name}/")
                        } else if let Some(size) = size {
                            format!("{name}  ({:.1} KB)", *size as f64 / 1024.0)
                        } else {
                            format!("{name}  (unreadable)")
                        };
                        if size.is_none() && !is_dir {
                            ui.text_disabled(label);
                            continue;
                        }
                        let selected =
                            !is_dir && *name == overlay_state.texture_fit_browser_selected;
                        let clicked = ui.selectable_config(&label).selected(selected).build();
                        let double_clicked = ui.is_item_hovered()
                            && ui.is_mouse_double_clicked(imgui::MouseButton::Left);
                        if *is_dir {
                            if double_clicked {
                                jump = Some(format!("{}/{}", dir_now.trim_end_matches('/'), name));
                            }
                        } else {
                            if clicked {
                                overlay_state.texture_fit_browser_selected = name.clone();
                            }
                            if double_clicked {
                                confirmed =
                                    Some(format!("{}/{}", dir_now.trim_end_matches('/'), name));
                            }
                        }
                    }
                    if truncated {
                        ui.text_colored(
                            [0.9, 0.7, 0.3, 1.0],
                            format!("listing capped at {TEXTURE_FIT_BROWSER_MAX_ENTRIES} entries"),
                        );
                    }
                });

            let has_selection = !overlay_state.texture_fit_browser_selected.is_empty();
            ui.enabled(has_selection, || {
                if ui.button("Open") {
                    confirmed = Some(format!(
                        "{}/{}",
                        dir_now.trim_end_matches('/'),
                        overlay_state.texture_fit_browser_selected
                    ));
                }
            });
            ui.same_line();
            if ui.button("Cancel") {
                overlay_state.texture_fit_browser_open = false;
            }

            if let Some(target) = jump {
                overlay_state.texture_fit_browser_dir = canonical_dir_or(&target);
                overlay_state.texture_fit_browser_selected.clear();
            }
        });
    if let Some(path) = confirmed {
        overlay_state.texture_fit_path = path;
        overlay_state.texture_fit_browser_open = false;
    }
    if !open {
        overlay_state.texture_fit_browser_open = false;
    }
}
