//! Resolve a router decision into an action: dispatch, await confirmation, or feedback.

use crate::helm::components::route::{Route, SlotKind};
use crate::helm::components::tool_call::{RiskLevel, ToolCall};

use super::binder::{bind_route, BindOutcome};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfirmReason {
    ConfirmAll,
    NearMiss,
    LowConfidence,
    Mutating,
}

/// Determine whether a tool call needs user confirmation and why.
///
/// Priority:
/// - If `confirm_all` is true -> `Some(ConfirmAll)`
/// - Else if `raw_near_miss` is true -> `Some(NearMiss)`
/// - Else if `needs_confirm` is true -> `Some(LowConfidence)`
/// - Else if `call.risk_level()` is `RiskLevel::Mutating` or `RiskLevel::Destructive` -> `Some(Mutating)`
/// - Else -> `None`
pub fn confirm_reason(call: &ToolCall, needs_confirm: bool, raw_near_miss: bool, confirm_all: bool) -> Option<ConfirmReason> {
    if confirm_all {
        Some(ConfirmReason::ConfirmAll)
    } else if raw_near_miss {
        Some(ConfirmReason::NearMiss)
    } else if needs_confirm {
        Some(ConfirmReason::LowConfidence)
    } else if matches!(call.risk_level(), RiskLevel::Mutating | RiskLevel::Destructive) {
        Some(ConfirmReason::Mutating)
    } else {
        None
    }
}

#[derive(Clone, Debug)]
pub enum ResolvedAction {
    Dispatch(ToolCall),
    AwaitConfirm {
        call: ToolCall,
        reason: ConfirmReason,
    },
    Feedback(HelmFeedback),
}

#[derive(Clone, Debug)]
pub enum HelmFeedback {
    Rejected {
        best: Route,
        score: f32,
    },
    ClarifyOptions(Vec<(Route, f32)>),
    MissingObjectName {
        route: Route,
    },
    AmbiguousObjectName {
        candidates: Vec<String>,
    },
    NoCandidate,
}

use crate::helm::systems::router::RouterDecision;

/// Resolve a router decision into a concrete action.
pub fn resolve_decision(
    decision: RouterDecision,
    normalized_utterance: &str,
    scene_names: &[String],
    confirm_all: bool,
) -> ResolvedAction {
    match decision {
        RouterDecision::Accept { route, score: _, needs_confirm, raw_near_miss } => {
            match bind_route(route.clone(), normalized_utterance, scene_names) {
                BindOutcome::Call(call) => {
                    if let Some(reason) = confirm_reason(&call, needs_confirm, raw_near_miss, confirm_all) {
                        ResolvedAction::AwaitConfirm { call, reason }
                    } else {
                        ResolvedAction::Dispatch(call)
                    }
                }
                BindOutcome::MissingSlot { route, slot: _ } => {
                    ResolvedAction::Feedback(HelmFeedback::MissingObjectName { route })
                }
                BindOutcome::AmbiguousSlot { route: _, candidates } => {
                    ResolvedAction::Feedback(HelmFeedback::AmbiguousObjectName { candidates })
                }
            }
        }
        RouterDecision::Clarify { candidates } => {
            ResolvedAction::Feedback(HelmFeedback::ClarifyOptions(candidates))
        }
        RouterDecision::Reject { best, score } => {
            ResolvedAction::Feedback(HelmFeedback::Rejected { best, score })
        }
        RouterDecision::NoCandidate => {
            ResolvedAction::Feedback(HelmFeedback::NoCandidate)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helm::components::tool_call::{MotionCategory, SpeedPreset};

    #[test]
    fn confirm_reason_confirm_all_priority() {
        let call = ToolCall::ListObjects;
        assert_eq!(
            confirm_reason(&call, false, false, true),
            Some(ConfirmReason::ConfirmAll)
        );
    }

    #[test]
    fn confirm_reason_near_miss_priority() {
        let call = ToolCall::ListObjects;
        assert_eq!(
            confirm_reason(&call, false, true, false),
            Some(ConfirmReason::NearMiss)
        );
    }

    #[test]
    fn confirm_reason_low_confidence() {
        let call = ToolCall::ListObjects;
        assert_eq!(
            confirm_reason(&call, true, false, false),
            Some(ConfirmReason::LowConfidence)
        );
    }

    #[test]
    fn confirm_reason_none_for_read_only() {
        let call = ToolCall::ListObjects;
        assert_eq!(confirm_reason(&call, false, false, false), None);
    }

    #[test]
    fn confirm_reason_mutating_for_play_animation() {
        let call = ToolCall::PlayAnimation;
        assert_eq!(
            confirm_reason(&call, false, false, false),
            Some(ConfirmReason::Mutating)
        );
    }

    #[test]
    fn resolve_accept_dispatch_read_only_no_confirm() {
        let decision = RouterDecision::Accept {
            route: Route::ListObjects,
            score: 0.9,
            needs_confirm: false,
            raw_near_miss: false,
        };
        let action = resolve_decision(decision, "list objects", &[], false);
        match action {
            ResolvedAction::Dispatch(call) => {
                assert_eq!(call, ToolCall::ListObjects);
            }
            other => panic!("expected Dispatch, got {:?}", other),
        }
    }

    #[test]
    fn resolve_accept_await_confirm_when_confirm_all() {
        let decision = RouterDecision::Accept {
            route: Route::ListObjects,
            score: 0.9,
            needs_confirm: false,
            raw_near_miss: false,
        };
        let action = resolve_decision(decision, "list objects", &[], true);
        match action {
            ResolvedAction::AwaitConfirm { call, reason } => {
                assert_eq!(call, ToolCall::ListObjects);
                assert_eq!(reason, ConfirmReason::ConfirmAll);
            }
            other => panic!("expected AwaitConfirm, got {:?}", other),
        }
    }

    #[test]
    fn resolve_accept_missing_object_name() {
        let decision = RouterDecision::Accept {
            route: Route::SelectObject,
            score: 0.9,
            needs_confirm: false,
            raw_near_miss: false,
        };
        let action = resolve_decision(decision, "select the lamp", &["Hero".to_string()], false);
        match action {
            ResolvedAction::Feedback(HelmFeedback::MissingObjectName { route }) => {
                assert_eq!(route, Route::SelectObject);
            }
            other => panic!("expected MissingObjectName feedback, got {:?}", other),
        }
    }

    #[test]
    fn resolve_clarify_feedback() {
        let candidates = vec![(Route::ListObjects, 0.5), (Route::DescribeSelection, 0.4)];
        let decision = RouterDecision::Clarify { candidates: candidates.clone() };
        let action = resolve_decision(decision, "what?", &[], false);
        match action {
            ResolvedAction::Feedback(HelmFeedback::ClarifyOptions(opts)) => {
                assert_eq!(opts, candidates);
            }
            other => panic!("expected ClarifyOptions feedback, got {:?}", other),
        }
    }

    #[test]
    fn resolve_reject_feedback() {
        let decision = RouterDecision::Reject { best: Route::ListObjects, score: 0.3 };
        let action = resolve_decision(decision, "nonsense", &[], false);
        match action {
            ResolvedAction::Feedback(HelmFeedback::Rejected { best, score }) => {
                assert_eq!(best, Route::ListObjects);
                assert!((score - 0.3).abs() < 1e-6);
            }
            other => panic!("expected Rejected feedback, got {:?}", other),
        }
    }

    #[test]
    fn resolve_no_candidate_feedback() {
        let decision = RouterDecision::NoCandidate;
        let action = resolve_decision(decision, "whatever", &[], false);
        match action {
            ResolvedAction::Feedback(HelmFeedback::NoCandidate) => {}
            other => panic!("expected NoCandidate feedback, got {:?}", other),
        }
    }
}
