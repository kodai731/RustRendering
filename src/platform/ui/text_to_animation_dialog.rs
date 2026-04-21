use crate::ecs::events::UIEventQueue;
use crate::ecs::resource::{TextToAnimationState, TextToAnimationStatus};
use crate::ecs::World;

pub struct TextToAnimationDialogState {
    pub open: bool,
    pub prompt_buf: String,
    pub duration: f32,
}

impl Default for TextToAnimationDialogState {
    fn default() -> Self {
        Self {
            open: false,
            prompt_buf: String::new(),
            duration: 3.0,
        }
    }
}

pub fn build_text_to_animation_dialog(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    dialog: &mut TextToAnimationDialogState,
    world: &World,
) {
    if !dialog.open {
        return;
    }

    if !world.contains_resource::<TextToAnimationState>() {
        return;
    }

    let snapshot = snapshot_state(world);

    let mut should_close = false;

    ui.window("Text to Animation")
        .size([420.0, 360.0], imgui::Condition::FirstUseEver)
        .build(|| {
            build_input_section(
                ui,
                ui_events,
                &mut dialog.prompt_buf,
                &mut dialog.duration,
                &snapshot,
                &mut should_close,
            );
            ui.separator();
            build_status_section(ui, &snapshot);
        });

    if should_close {
        dialog.open = false;
    }
}

struct StateSnapshot {
    status: TextToAnimationStatus,
    error_message: Option<String>,
    generation_time_ms: Option<f32>,
    model_used: Option<String>,
    skipped_auto_rig: bool,
}

fn snapshot_state(world: &World) -> StateSnapshot {
    let state = world.resource::<TextToAnimationState>();
    StateSnapshot {
        status: state.status.clone(),
        error_message: state.error_message.clone(),
        generation_time_ms: state.generation_time_ms,
        model_used: state.model_used.clone(),
        skipped_auto_rig: state.skipped_auto_rig,
    }
}

fn is_in_progress(status: &TextToAnimationStatus) -> bool {
    matches!(
        status,
        TextToAnimationStatus::AutoRigging
            | TextToAnimationStatus::AutoRigApplying
            | TextToAnimationStatus::GeneratingMotion
            | TextToAnimationStatus::ApplyingClip
    )
}

fn build_input_section(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    prompt_buf: &mut String,
    duration: &mut f32,
    snapshot: &StateSnapshot,
    should_close: &mut bool,
) {
    let in_progress = is_in_progress(&snapshot.status);

    ui.text("Prompt:");
    ui.input_text("##prompt", prompt_buf)
        .hint("e.g. walking forward slowly")
        .build();

    ui.text("Duration (sec):");
    ui.same_line();
    ui.set_next_item_width(100.0);
    imgui::Drag::new("##duration")
        .range(0.5, 10.0)
        .speed(0.1)
        .display_format("%.1f")
        .build(ui, duration);

    let can_generate = !in_progress && !prompt_buf.trim().is_empty();

    ui.spacing();
    if in_progress {
        ui.text("Working...");
    } else {
        let _disabled = ui.begin_disabled(!can_generate);
        if ui.button("Generate") {
            ui_events.send(crate::ecs::events::UIEvent::TextToAnimationGenerate {
                prompt: prompt_buf.trim().to_string(),
                duration_seconds: *duration,
            });
        }
    }

    ui.same_line();
    if ui.button("Cancel") {
        if in_progress {
            ui_events.send(crate::ecs::events::UIEvent::TextToAnimationCancel);
        } else {
            *should_close = true;
        }
    }
}

fn build_status_section(ui: &imgui::Ui, snapshot: &StateSnapshot) {
    let status_text = match snapshot.status {
        TextToAnimationStatus::Idle => "Idle",
        TextToAnimationStatus::AutoRigging => "Auto-rigging...",
        TextToAnimationStatus::AutoRigApplying => "Loading rigged model...",
        TextToAnimationStatus::GeneratingMotion => "Generating motion...",
        TextToAnimationStatus::ApplyingClip => "Applying clip...",
        TextToAnimationStatus::Done => "Done",
        TextToAnimationStatus::Error => "Error",
    };
    ui.text(format!("Status: {}", status_text));

    if snapshot.skipped_auto_rig
        && !matches!(
            snapshot.status,
            TextToAnimationStatus::Idle | TextToAnimationStatus::Error
        )
    {
        ui.text_colored(
            [0.6, 0.8, 1.0, 1.0],
            "Skin weights detected, auto-rig skipped",
        );
    }

    if let Some(time_ms) = snapshot.generation_time_ms {
        ui.text(format!("Motion generation: {:.0}ms", time_ms));
    }

    if let Some(model) = &snapshot.model_used {
        ui.text(format!("Model: {}", model));
    }

    if let Some(err) = &snapshot.error_message {
        ui.text_colored([1.0, 0.3, 0.3, 1.0], format!("Error: {}", err));
    }
}
