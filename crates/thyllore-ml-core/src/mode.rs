use crate::copilot::v2::inference::CONTEXT_LENGTH;
use crate::degrade::DEGRADED_CONTEXT_LENGTH;

/// Distribution mode SSoT for Curve Copilot. The engine launch flag
/// (`--curve-copilot full|degrade|private`) and the Blender addon build mode
/// (`scripts/build_blender_addon.{sh,ps1} --build-mode A|B|C`) select the
/// same three paths:
///
/// | Mode    | Addon build          | Behaviour                     | Extra build env                                           |
/// |---------|----------------------|-------------------------------|-----------------------------------------------------------|
/// | Degrade | A (official repo)    | ctx32 fixed, no data sending  | (none)                                                    |
/// | Full    | B (self-hosted repo) | ctx64, sends feedback records | `THYLLORE_FULL_TOKEN_PUBKEY_B64`                              |
/// | Private | C (Blender Market)   | ctx64 via license, no records | `THYLLORE_LICENSE_ENDPOINT`, `THYLLORE_FULL_TOKEN_PUBKEY_B64` |
///
/// Every build mode requires `THYLLORE_FEEDBACK_ENDPOINT` and
/// `THYLLORE_INGEST_TOKEN` for the free-text message channel (`/v1/message`),
/// which is available in all modes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CurveCopilotMode {
    Full,
    Degrade,
    #[default]
    Private,
}

impl CurveCopilotMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "full" => Some(Self::Full),
            "degrade" => Some(Self::Degrade),
            "private" => Some(Self::Private),
            _ => None,
        }
    }

    pub fn effective_context_length(self) -> usize {
        match self {
            Self::Full | Self::Private => CONTEXT_LENGTH,
            Self::Degrade => DEGRADED_CONTEXT_LENGTH,
        }
    }

    pub fn sends_feedback(self) -> bool {
        matches!(self, Self::Full)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_known_modes() {
        assert_eq!(
            CurveCopilotMode::parse("full"),
            Some(CurveCopilotMode::Full)
        );
        assert_eq!(
            CurveCopilotMode::parse("degrade"),
            Some(CurveCopilotMode::Degrade)
        );
        assert_eq!(
            CurveCopilotMode::parse("private"),
            Some(CurveCopilotMode::Private)
        );
    }

    #[test]
    fn parse_rejects_unknown_values() {
        assert_eq!(CurveCopilotMode::parse("Full"), None);
        assert_eq!(CurveCopilotMode::parse(""), None);
        assert_eq!(CurveCopilotMode::parse("ctx32"), None);
    }

    #[test]
    fn default_is_private() {
        assert_eq!(CurveCopilotMode::default(), CurveCopilotMode::Private);
    }

    #[test]
    fn only_degrade_shortens_context() {
        assert_eq!(
            CurveCopilotMode::Full.effective_context_length(),
            CONTEXT_LENGTH
        );
        assert_eq!(
            CurveCopilotMode::Private.effective_context_length(),
            CONTEXT_LENGTH
        );
        assert_eq!(
            CurveCopilotMode::Degrade.effective_context_length(),
            DEGRADED_CONTEXT_LENGTH
        );
    }

    #[test]
    fn only_full_sends_feedback() {
        assert!(CurveCopilotMode::Full.sends_feedback());
        assert!(!CurveCopilotMode::Degrade.sends_feedback());
        assert!(!CurveCopilotMode::Private.sends_feedback());
    }
}
