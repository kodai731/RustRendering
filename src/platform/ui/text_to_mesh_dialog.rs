use crate::ecs::events::UIEventQueue;
use crate::ecs::resource::{TextToMeshState, TextToMeshStatus};
use crate::ecs::World;
use crate::grpc::MeshInputMode;

pub struct TextToMeshDialogState {
    pub open: bool,
    pub prompt_buf: String,
    pub target_faces: i32,
    pub seed: i32,
    pub generate_start_time: Option<std::time::Instant>,
    pub input_mode: MeshInputMode,
    pub image_path: String,
    pub image_bytes: Option<Vec<u8>>,
    pub image_load_error: Option<String>,
}

impl Default for TextToMeshDialogState {
    fn default() -> Self {
        Self {
            open: false,
            prompt_buf: String::new(),
            target_faces: 50000,
            seed: 0,
            generate_start_time: None,
            input_mode: MeshInputMode::TextOnly,
            image_path: String::new(),
            image_bytes: None,
            image_load_error: None,
        }
    }
}

pub fn build_text_to_mesh_dialog(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    dialog: &mut TextToMeshDialogState,
    world: &World,
) {
    if !dialog.open {
        return;
    }

    if !world.contains_resource::<TextToMeshState>() {
        return;
    }

    let state = world.resource::<TextToMeshState>();
    let status = state.status.clone();
    let error_msg = state.error_message.clone();
    let gen_time = state.generation_time_ms;
    let vertex_count = state.vertex_count;
    let face_count = state.face_count;
    let has_glb = state.glb_data.is_some();
    drop(state);

    let mut should_close = false;

    ui.window("Mesh Generation")
        .size([420.0, 440.0], imgui::Condition::FirstUseEver)
        .build(|| {
            build_mode_tabs(ui, dialog);
            ui.separator();
            build_input_section(ui, ui_events, dialog, &status, &mut should_close);
            ui.separator();
            build_status_section(ui, &status, &error_msg, gen_time, dialog);

            if has_glb {
                ui.separator();
                build_result_section(ui, ui_events, vertex_count, face_count, &mut should_close);
            }
        });

    if should_close {
        dialog.open = false;
        dialog.generate_start_time = None;
    }
}

fn build_mode_tabs(ui: &imgui::Ui, dialog: &mut TextToMeshDialogState) {
    imgui::TabBar::new("##mesh_mode_tabs").build(ui, || {
        imgui::TabItem::new("Text").build(ui, || {
            dialog.input_mode = MeshInputMode::TextOnly;
        });

        imgui::TabItem::new("Image").build(ui, || {
            dialog.input_mode = MeshInputMode::Image;
        });
    });
}

fn build_input_section(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    dialog: &mut TextToMeshDialogState,
    status: &TextToMeshStatus,
    should_close: &mut bool,
) {
    let is_busy =
        *status == TextToMeshStatus::Generating || *status == TextToMeshStatus::WaitingForServer;

    if dialog.input_mode == MeshInputMode::Image {
        build_image_picker(ui, dialog);
        ui.spacing();
    }

    if dialog.input_mode == MeshInputMode::TextOnly {
        ui.text("Prompt:");
    } else {
        ui.text("Description (optional):");
    }
    ui.input_text("##mesh_prompt", &mut dialog.prompt_buf)
        .hint("e.g. a cute robot character")
        .build();

    ui.text("Target Faces:");
    ui.same_line();
    ui.set_next_item_width(150.0);
    imgui::Drag::new("##target_faces")
        .range(1000, 200000)
        .speed(1000.0)
        .build(ui, &mut dialog.target_faces);

    ui.text("Seed:");
    ui.same_line();
    ui.set_next_item_width(100.0);
    imgui::Drag::new("##seed")
        .range(0, 999999)
        .speed(1.0)
        .build(ui, &mut dialog.seed);
    ui.same_line();
    ui.text_disabled("(0 = random)");

    let can_generate = !is_busy
        && match dialog.input_mode {
            MeshInputMode::TextOnly => !dialog.prompt_buf.trim().is_empty(),
            MeshInputMode::Image => dialog.image_bytes.is_some(),
        };

    ui.spacing();
    if is_busy {
        match status {
            TextToMeshStatus::WaitingForServer => ui.text("Waiting for server..."),
            _ => ui.text("Generating..."),
        }
    } else {
        let _disabled = ui.begin_disabled(!can_generate);
        if ui.button("Generate") {
            ui_events.send(crate::ecs::events::UIEvent::TextToMeshGenerate {
                prompt: dialog.prompt_buf.trim().to_string(),
                target_faces: dialog.target_faces as u32,
                seed: dialog.seed as u32,
                input_mode: dialog.input_mode.clone(),
                input_image_png: dialog.image_bytes.clone(),
            });
            dialog.generate_start_time = Some(std::time::Instant::now());
        }
    }

    ui.same_line();
    if ui.button("Cancel") {
        if is_busy {
            ui_events.send(crate::ecs::events::UIEvent::TextToMeshCancel);
            dialog.generate_start_time = None;
        } else {
            *should_close = true;
        }
    }
}

fn build_image_picker(ui: &imgui::Ui, dialog: &mut TextToMeshDialogState) {
    ui.text("Image (PNG):");
    ui.same_line();

    if ui.button("Browse...") {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("PNG Image", &["png"])
            .pick_file()
        {
            let path_str = path.to_string_lossy().to_string();
            match std::fs::read(&path) {
                Ok(bytes) => {
                    dialog.image_path = path_str;
                    dialog.image_bytes = Some(bytes);
                    dialog.image_load_error = None;
                }
                Err(e) => {
                    dialog.image_path = path_str;
                    dialog.image_bytes = None;
                    dialog.image_load_error = Some(format!("Failed to read: {}", e));
                }
            }
        }
    }

    if !dialog.image_path.is_empty() {
        let display_name = std::path::Path::new(&dialog.image_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&dialog.image_path);
        ui.text(display_name);

        if let Some(ref bytes) = dialog.image_bytes {
            ui.same_line();
            ui.text_disabled(format!("({} KB)", bytes.len() / 1024));
        }
    }

    if let Some(ref err) = dialog.image_load_error {
        ui.text_colored([1.0, 0.3, 0.3, 1.0], err);
    }
}

fn build_status_section(
    ui: &imgui::Ui,
    status: &TextToMeshStatus,
    error_msg: &Option<String>,
    gen_time: Option<f32>,
    dialog: &TextToMeshDialogState,
) {
    let status_text = match status {
        TextToMeshStatus::Idle => "Idle".to_string(),
        TextToMeshStatus::WaitingForServer => {
            if let Some(start) = dialog.generate_start_time {
                let elapsed = start.elapsed().as_secs();
                format!("Waiting for server... ({}s)", elapsed)
            } else {
                "Waiting for server...".to_string()
            }
        }
        TextToMeshStatus::Generating => {
            if let Some(start) = dialog.generate_start_time {
                let elapsed = start.elapsed().as_secs();
                format!("Generating... ({}s)", elapsed)
            } else {
                "Generating...".to_string()
            }
        }
        TextToMeshStatus::Generated => "Generated".to_string(),
        TextToMeshStatus::Error => "Error".to_string(),
    };
    ui.text(format!("Status: {}", status_text));

    if let Some(time_ms) = gen_time {
        ui.text(format!("Generation time: {:.1}s", time_ms / 1000.0));
    }

    if let Some(err) = error_msg {
        ui.text_colored([1.0, 0.3, 0.3, 1.0], format!("Error: {}", err));
    }
}

fn build_result_section(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    vertex_count: Option<u32>,
    face_count: Option<u32>,
    should_close: &mut bool,
) {
    if let Some(verts) = vertex_count {
        ui.text(format!("Vertices: {}", verts));
    }
    if let Some(faces) = face_count {
        ui.text(format!("Faces: {}", faces));
    }
    ui.spacing();

    if ui.button("Apply to Scene") {
        ui_events.send(crate::ecs::events::UIEvent::TextToMeshApply);
        *should_close = true;
    }

    ui.same_line();
    if ui.button("Dismiss") {
        ui_events.send(crate::ecs::events::UIEvent::TextToMeshCancel);
    }
}
