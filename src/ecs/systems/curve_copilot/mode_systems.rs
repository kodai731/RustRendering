use anyhow::{anyhow, bail, Result};
use thyllore_ml_core::copilot::v2::inference::CONTEXT_LENGTH;
use thyllore_ml_core::degrade::degrade_context_window;

use crate::animation::editable::PropertyType;
use crate::ml::{build_engine_feedback_record, CurveCopilotMode, FeedbackSenderHandle};

pub const CURVE_COPILOT_MODE_FLAG: &str = "--curve-copilot";
pub const CURVE_COPILOT_MODE_ENV: &str = "THYLLORE_CURVE_COPILOT_MODE";

pub fn curve_copilot_mode_resolve_from_env_args() -> Result<CurveCopilotMode> {
    let args: Vec<String> = std::env::args().collect();
    let env_value = std::env::var(CURVE_COPILOT_MODE_ENV).ok();
    curve_copilot_mode_resolve(&args, env_value.as_deref())
}

pub fn curve_copilot_mode_resolve(
    args: &[String],
    env_value: Option<&str>,
) -> Result<CurveCopilotMode> {
    if let Some(position) = args.iter().position(|arg| arg == CURVE_COPILOT_MODE_FLAG) {
        let Some(value) = args.get(position + 1) else {
            bail!("{CURVE_COPILOT_MODE_FLAG} requires a value: full|degrade|private");
        };
        return parse_mode(value);
    }

    if let Some(value) = env_value {
        return parse_mode(value);
    }
    Ok(CurveCopilotMode::default())
}

fn parse_mode(value: &str) -> Result<CurveCopilotMode> {
    CurveCopilotMode::parse(value).ok_or_else(|| {
        anyhow!("invalid curve copilot mode '{value}': expected full|degrade|private")
    })
}

pub fn curve_copilot_degrade_context(mode: CurveCopilotMode, context: &mut [f32]) {
    if mode.effective_context_length() < CONTEXT_LENGTH {
        degrade_context_window(context);
    }
}

pub fn curve_copilot_capture_feedback_context(
    mode: CurveCopilotMode,
    context: &[f32],
) -> Option<Vec<f32>> {
    if mode.sends_feedback() {
        Some(context.to_vec())
    } else {
        None
    }
}

pub fn curve_copilot_send_feedback(
    sender: &FeedbackSenderHandle,
    property_type: PropertyType,
    origin_value: f32,
    context: &[f32],
    mean_curve: &[f32],
) {
    let record = build_engine_feedback_record(
        sender.model_hash(),
        property_type,
        origin_value,
        context,
        mean_curve,
    );
    sender.send(record);
    log!(
        "CurveCopilot feedback: record queued (channel={:?})",
        property_type
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_ml_core::degrade::DEGRADED_CONTEXT_LENGTH;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    #[test]
    fn defaults_to_private_without_flag_or_env() {
        let mode = curve_copilot_mode_resolve(&args(&["engine"]), None).unwrap();
        assert_eq!(mode, CurveCopilotMode::Private);
    }

    #[test]
    fn flag_selects_mode() {
        let mode =
            curve_copilot_mode_resolve(&args(&["engine", "--curve-copilot", "degrade"]), None)
                .unwrap();
        assert_eq!(mode, CurveCopilotMode::Degrade);
    }

    #[test]
    fn flag_takes_precedence_over_env() {
        let mode = curve_copilot_mode_resolve(
            &args(&["engine", "--curve-copilot", "full"]),
            Some("degrade"),
        )
        .unwrap();
        assert_eq!(mode, CurveCopilotMode::Full);
    }

    #[test]
    fn env_selects_mode_without_flag() {
        let mode = curve_copilot_mode_resolve(&args(&["engine"]), Some("full")).unwrap();
        assert_eq!(mode, CurveCopilotMode::Full);
    }

    #[test]
    fn flag_without_value_fails() {
        assert!(curve_copilot_mode_resolve(&args(&["engine", "--curve-copilot"]), None).is_err());
    }

    #[test]
    fn invalid_mode_fails() {
        assert!(
            curve_copilot_mode_resolve(&args(&["engine", "--curve-copilot", "ctx32"]), None)
                .is_err()
        );
    }

    #[test]
    fn degrade_mode_zero_pads_context() {
        let mut context: Vec<f32> = (1..=CONTEXT_LENGTH).map(|v| v as f32).collect();
        curve_copilot_degrade_context(CurveCopilotMode::Degrade, &mut context);

        let pad = CONTEXT_LENGTH - DEGRADED_CONTEXT_LENGTH;
        assert!(context[..pad].iter().all(|&v| v == 0.0));
        assert!(context[pad..].iter().all(|&v| v != 0.0));
    }

    #[test]
    fn full_and_private_keep_context() {
        for mode in [CurveCopilotMode::Full, CurveCopilotMode::Private] {
            let mut context: Vec<f32> = (1..=CONTEXT_LENGTH).map(|v| v as f32).collect();
            curve_copilot_degrade_context(mode, &mut context);
            assert!(context.iter().all(|&v| v != 0.0));
        }
    }

    #[test]
    fn only_full_captures_feedback_context() {
        let context = vec![1.0_f32, 2.0];
        assert_eq!(
            curve_copilot_capture_feedback_context(CurveCopilotMode::Full, &context),
            Some(context.clone())
        );
        assert_eq!(
            curve_copilot_capture_feedback_context(CurveCopilotMode::Degrade, &context),
            None
        );
        assert_eq!(
            curve_copilot_capture_feedback_context(CurveCopilotMode::Private, &context),
            None
        );
    }
}
