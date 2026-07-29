use imgui::{Condition, Ui};

use crate::ecs::World;
use crate::helm::components::route::HelmMode;

#[derive(Clone, Debug)]
pub struct CommandBarState {
    pub open: bool,
    pub input_buffer: String,
}

impl Default for CommandBarState {
    fn default() -> Self {
        Self {
            open: true,
            input_buffer: String::new(),
        }
    }
}

fn feedback_text(feedback: &crate::ecs::resource::CommandFeedback) -> String {
    match feedback {
        crate::ecs::resource::CommandFeedback::Report(report) => report.clone(),
        crate::ecs::resource::CommandFeedback::Executed(report) => format!("実行済み: {}", report),
        crate::ecs::resource::CommandFeedback::DispatchError(err) => format!("エラー: {}", err),
        crate::ecs::resource::CommandFeedback::Unavailable(s) => s.clone(),
        crate::ecs::resource::CommandFeedback::Router(orch_feedback) => {
            match orch_feedback {
                crate::helm::systems::resolution::HelmFeedback::Rejected { best, .. } => {
                    format!("未実行: {:?} の可能性 — 言い換えるか候補を確認してください", best)
                }
                crate::helm::systems::resolution::HelmFeedback::ClarifyOptions(_) => {
                    "複数の候補があります。下のボタンから選択してください".to_string()
                }
                crate::helm::systems::resolution::HelmFeedback::MissingObjectName { .. } => {
                    "対象オブジェクト名が見つかりません。名前を含めて言い直してください".to_string()
                }
                crate::helm::systems::resolution::HelmFeedback::AmbiguousObjectName { candidates } => {
                    format!("曖昧な名前: {:?} — 特定の名前を指定してください", candidates)
                }
                crate::helm::systems::resolution::HelmFeedback::NoCandidate => {
                    "一致する候補が見つかりませんでした".to_string()
                }
            }
        }
    }
}

fn confirm_reason_text(reason: &crate::helm::systems::resolution::ConfirmReason) -> &'static str {
    match reason {
        crate::helm::systems::resolution::ConfirmReason::ConfirmAll => "確認モード",
        crate::helm::systems::resolution::ConfirmReason::NearMiss => "類似度が低め (要確認)",
        crate::helm::systems::resolution::ConfirmReason::LowConfidence => "低信頼度",
        crate::helm::systems::resolution::ConfirmReason::Mutating => "変更操作",
    }
}

pub fn build_command_bar(ui: &Ui, state_ui: &mut CommandBarState, world: &World) {
    let mut helm_state = match world.get_resource_mut::<crate::ecs::resource::HelmState>() {
        Some(r) => r,
        None => return,
    };

    ui.open_popup("command_bar_popup");

    // ① Input
    let mut buffer = state_ui.input_buffer.clone();
    let sent = ui
        .input_text("##command", &mut buffer)
        .enter_returns_true(true)
        .build()
        || ui.small_button("Send");

    if sent && !buffer.is_empty() {
        helm_state.submitted_utterance = Some(buffer.clone());
        state_ui.input_buffer.clear();
    } else {
        state_ui.input_buffer = buffer;
    }

    // ② Feedback display
    if let Some(ref feedback) = helm_state.feedback {
        ui.text(feedback_text(feedback));
    }

    // ③ ClarifyOptions: candidate buttons
    let options_snapshot = if let Some(crate::ecs::resource::CommandFeedback::Router(
        crate::helm::systems::resolution::HelmFeedback::ClarifyOptions(options),
    )) = &helm_state.feedback {
        Some(options.clone())
    } else {
        None
    };
    if let Some(options) = options_snapshot {
        for (route, score) in options.iter() {
            let route_copy = *route;
            if ui.small_button(format!("{:?} ({:.2})", route_copy, score)) {
                helm_state.clarify_choice = Some(route_copy);
            }
        }
    }

    // ④ Pending confirmation
    if let Some((ref call, ref reason)) = helm_state.pending {
        ui.separator();
        ui.text(format!("ツール: {}", call.tool_name()));
        ui.text(format!("理由: {}", confirm_reason_text(reason)));
        ui.separator();

        if ui.small_button("Execute") {
            helm_state.confirm_response = Some(true);
        }
        ui.same_line();
        if ui.small_button("Cancel") {
            helm_state.confirm_response = Some(false);
        }
    }

    // ⑤ Read Only / Allow Edit toggle
    let mut allow = matches!(helm_state.mode, HelmMode::AllowEdit);
    if ui.checkbox("Allow Edit", &mut allow) {
        helm_state.mode = if allow {
            HelmMode::AllowEdit
        } else {
            HelmMode::ReadOnly
        };
    }

    ui.close_current_popup();
}
