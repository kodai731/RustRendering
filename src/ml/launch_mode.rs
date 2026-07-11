use anyhow::{anyhow, bail, Result};
use thyllore_ml_core::CurveCopilotMode;

pub const CURVE_COPILOT_MODE_FLAG: &str = "--curve-copilot";
pub const CURVE_COPILOT_MODE_ENV: &str = "THYLLORE_CURVE_COPILOT_MODE";

pub fn resolve_curve_copilot_mode_from_env_args() -> Result<CurveCopilotMode> {
    let args: Vec<String> = std::env::args().collect();
    let env_value = std::env::var(CURVE_COPILOT_MODE_ENV).ok();
    resolve_curve_copilot_mode(&args, env_value.as_deref())
}

fn resolve_curve_copilot_mode(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    #[test]
    fn defaults_to_private_without_flag_or_env() {
        let mode = resolve_curve_copilot_mode(&args(&["engine"]), None).unwrap();
        assert_eq!(mode, CurveCopilotMode::Private);
    }

    #[test]
    fn flag_selects_mode() {
        let mode =
            resolve_curve_copilot_mode(&args(&["engine", "--curve-copilot", "degrade"]), None)
                .unwrap();
        assert_eq!(mode, CurveCopilotMode::Degrade);
    }

    #[test]
    fn flag_takes_precedence_over_env() {
        let mode = resolve_curve_copilot_mode(
            &args(&["engine", "--curve-copilot", "full"]),
            Some("degrade"),
        )
        .unwrap();
        assert_eq!(mode, CurveCopilotMode::Full);
    }

    #[test]
    fn env_selects_mode_without_flag() {
        let mode = resolve_curve_copilot_mode(&args(&["engine"]), Some("full")).unwrap();
        assert_eq!(mode, CurveCopilotMode::Full);
    }

    #[test]
    fn flag_without_value_fails() {
        assert!(resolve_curve_copilot_mode(&args(&["engine", "--curve-copilot"]), None).is_err());
    }

    #[test]
    fn invalid_mode_fails() {
        assert!(
            resolve_curve_copilot_mode(&args(&["engine", "--curve-copilot", "ctx32"]), None)
                .is_err()
        );
    }
}
