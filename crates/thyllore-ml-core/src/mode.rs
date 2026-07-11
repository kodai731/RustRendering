use crate::copilot::v2::inference::CONTEXT_LENGTH;
use crate::unlock::DEGRADED_CONTEXT_LENGTH;

/// Engine launch mode selecting the distribution behaviour on a single binary.
/// Bypasses the unlock-token gate and reproduces only its outcome; intended
/// for verifying the distribution paths (A=degrade, B=full, C/default=private).
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
