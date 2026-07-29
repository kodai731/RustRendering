//! Deterministic slot-filling binder: turns a chosen `Route` into a `ToolCall`.
//!
//! For routes with no slot (e.g. `SetPlaybackSpeed`) the mapping is direct.
//! For routes carrying an `ObjectName` slot (`SelectObject`, `SetObjectVisibility`)
//! the scene names are matched by normalized substring inclusion with longest-match
//! tiebreaking.

use crate::helm::components::route::{Route, SlotKind};
use crate::helm::components::tool_call::{ObjectName, SpeedPreset, ToolCall};

use super::modifier::extract_speed_modifier;
use super::normalize::normalize_utterance;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindOutcome {
    Call(ToolCall),
    MissingSlot {
        route: Route,
        slot: SlotKind,
    },
    AmbiguousSlot {
        route: Route,
        candidates: Vec<String>,
    },
}

/// Bind a `Route` to a `ToolCall` by filling its slot from the utterance.
pub fn bind_route(route: Route, normalized_utterance: &str, scene_names: &[String]) -> BindOutcome {
    match route.slot() {
        None => direct_call(route, normalized_utterance),
        Some(SlotKind::ObjectName) => resolve_object_name(route, normalized_utterance, scene_names),
    }
}

fn direct_call(route: Route, normalized_utterance: &str) -> BindOutcome {
    let tool_call = match route {
        Route::ListObjects => ToolCall::ListObjects,
        Route::DescribeSelection => ToolCall::DescribeSelection,
        Route::GetPlaybackState => ToolCall::GetPlaybackState,
        Route::TakeScreenshot => ToolCall::TakeScreenshot,
        Route::PlayAnimation => ToolCall::PlayAnimation,
        Route::PauseAnimation => ToolCall::PauseAnimation,
        Route::StopAnimation => ToolCall::StopAnimation,
        Route::SetPlaybackSpeed(preset) => ToolCall::SetPlaybackSpeed(preset),
        Route::SeekTime(position) => ToolCall::SeekTime(position),
        Route::ToggleLoop => ToolCall::ToggleLoop,
        Route::Undo => ToolCall::Undo,
        Route::Redo => ToolCall::Redo,
        Route::SaveScene => ToolCall::SaveScene,
        Route::FocusCamera(target) => ToolCall::FocusCamera(target),
        Route::GenerateMotion(category) => {
            let speed = extract_speed_modifier(normalized_utterance).unwrap_or(SpeedPreset::Normal);
            ToolCall::GenerateMotion(category, speed)
        }
      // ObjectName slot routes are handled by resolve_object_name
        Route::SelectObject | Route::SetObjectVisibility(_) => unreachable!(
            "slot() returned None for {:?}, but it has an ObjectName slot",
            route
        ),
        Route::EscapeAnchor => unreachable!("EscapeAnchor should be rejected before binding"),
    };
    BindOutcome::Call(tool_call)
}

fn resolve_object_name(
    route: Route,
    normalized_utterance: &str,
    scene_names: &[String],
) -> BindOutcome {
    let mut hits: Vec<(usize, &String)> = Vec::new();

    for name in scene_names {
        let normalized_name = normalize_utterance(name);
        if normalized_utterance.contains(normalized_name.as_str()) {
            hits.push((normalized_name.len(), name));
        }
    }

    match hits.len() {
        0 => BindOutcome::MissingSlot {
            route,
            slot: SlotKind::ObjectName,
        },
        1 => {
            let original_name = hits[0].1;
            let tool_call = make_object_tool_call(route, original_name);
            BindOutcome::Call(tool_call)
        }
        _ => {
            let max_len = hits.iter().map(|(len, _)| *len).max().unwrap();
            let best: Vec<&String> = hits
                .iter()
                .filter(|(len, _)| *len == max_len)
                .map(|(_, name)| *name)
                .collect();

            if best.len() == 1 {
                let original_name = best[0];
                let tool_call = make_object_tool_call(route, original_name);
                BindOutcome::Call(tool_call)
            } else {
                let candidates: Vec<String> = best.iter().map(|s| s.to_string()).collect();
                BindOutcome::AmbiguousSlot { route, candidates }
            }
        }
    }
}

fn make_object_tool_call(route: Route, name: &str) -> ToolCall {
    match route {
        Route::SelectObject => ToolCall::SelectObject(ObjectName(name.to_string())),
        Route::SetObjectVisibility(state) => {
            ToolCall::SetObjectVisibility(ObjectName(name.to_string()), state)
        }
        _ => unreachable!("make_object_tool_call called for non-object route {:?}", route),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helm::components::tool_call::{MotionCategory, VisibilityState};

    fn bind(route: Route, utterance: &str, scene_names: &[&str]) -> BindOutcome {
        let names: Vec<String> = scene_names.iter().map(|s| s.to_string()).collect();
        bind_route(route, utterance, &names)
    }

    #[test]
    fn direct_mapping_set_playback_speed() {
        let outcome = bind(Route::SetPlaybackSpeed(SpeedPreset::Fast), "fast", &[]);
        assert_eq!(outcome, BindOutcome::Call(ToolCall::SetPlaybackSpeed(SpeedPreset::Fast)));
    }

    #[test]
    fn direct_mapping_play_animation() {
        let outcome = bind(Route::PlayAnimation, "play", &[]);
        assert_eq!(outcome, BindOutcome::Call(ToolCall::PlayAnimation));
    }

    #[test]
    fn generate_motion_without_speed_modifier_defaults_to_normal() {
        let outcome = bind(
            Route::GenerateMotion(MotionCategory::Walk),
            "generate a walk",
            &[],
        );
        assert_eq!(
            outcome,
            BindOutcome::Call(ToolCall::GenerateMotion(
                MotionCategory::Walk,
                SpeedPreset::Normal
            ))
        );
    }

    #[test]
    fn generate_motion_with_slow_modifier() {
        let outcome = bind(
            Route::GenerateMotion(MotionCategory::Run),
            "generate a slow run",
            &[],
        );
        assert_eq!(
            outcome,
            BindOutcome::Call(ToolCall::GenerateMotion(
                MotionCategory::Run,
                SpeedPreset::Slow
            ))
        );
    }

    #[test]
    fn generate_motion_with_fast_modifier() {
        let outcome = bind(
            Route::GenerateMotion(MotionCategory::Jump),
            "generate a fast jump",
            &[],
        );
        assert_eq!(
            outcome,
            BindOutcome::Call(ToolCall::GenerateMotion(
                MotionCategory::Jump,
                SpeedPreset::Fast
            ))
        );
    }

    #[test]
    fn select_object_longest_match_herosword() {
        let scene_names: &[&str] = &["Hero", "HeroSword", "Floor"];
        let outcome = bind(Route::SelectObject, "select the herosword", scene_names);
        assert_eq!(
            outcome,
            BindOutcome::Call(ToolCall::SelectObject(ObjectName("HeroSword".to_string())))
        );
    }

    #[test]
    fn select_object_shorter_match_hero() {
        let scene_names: &[&str] = &["Hero", "HeroSword", "Floor"];
        let outcome = bind(Route::SelectObject, "select the hero", scene_names);
        assert_eq!(
            outcome,
            BindOutcome::Call(ToolCall::SelectObject(ObjectName("Hero".to_string())))
        );
    }

    #[test]
    fn select_object_missing_slot() {
        let scene_names: &[&str] = &["Hero", "HeroSword", "Floor"];
        let outcome = bind(Route::SelectObject, "select the lamp", scene_names);
        assert_eq!(
            outcome,
            BindOutcome::MissingSlot {
                route: Route::SelectObject,
                slot: SlotKind::ObjectName,
            }
        );
    }

    #[test]
    fn select_object_case_insensitive() {
        let scene_names: &[&str] = &["Hero"];
        let outcome = bind(Route::SelectObject, "select hero", scene_names);
        assert_eq!(
            outcome,
            BindOutcome::Call(ToolCall::SelectObject(ObjectName("Hero".to_string())))
        );
    }

    #[test]
    fn set_object_visibility_with_match() {
        let scene_names: &[&str] = &["Hero", "Floor"];
        let outcome = bind(
            Route::SetObjectVisibility(VisibilityState::Hide),
            "hide the hero",
            scene_names,
        );
        assert_eq!(
            outcome,
            BindOutcome::Call(ToolCall::SetObjectVisibility(
                ObjectName("Hero".to_string()),
                VisibilityState::Hide
            ))
        );
    }

    #[test]
    fn ambiguous_slot_same_length() {
        // "ab" and "cd" both have length 2; utterance contains both
        let scene_names: &[&str] = &["Ab", "Cd"];
        let outcome = bind(Route::SelectObject, "select ab and cd", scene_names);
        match outcome {
            BindOutcome::AmbiguousSlot { route, candidates } => {
                assert_eq!(route, Route::SelectObject);
                assert_eq!(candidates.len(), 2);
                assert!(candidates.contains(&"Ab".to_string()));
                assert!(candidates.contains(&"Cd".to_string()));
            }
            other => panic!("expected AmbiguousSlot, got {:?}", other),
        }
    }
}
