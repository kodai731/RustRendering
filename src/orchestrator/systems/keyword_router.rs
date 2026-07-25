//! Deterministic keyword routing, run before the embedding router.
//!
//! It does three things: extract enum modifiers, drop edit routes from the
//! candidate set when the utterance only asks for information, and pick a route
//! outright when a high-precision keyword rule matches. The rules exist because
//! embedding similarity confuses antonyms that share a context — start/end,
//! show/hide, undo/redo, pause/stop — and no amount of example utterances
//! separates those in embedding space.
//!
//! The rule table lives in `../data/keyword_router_rules.json` so that this
//! module and `scripts/orchestrator_eval/keyword_router.py` are driven by one
//! file. Measured numbers would not describe the shipped behaviour otherwise.

use std::sync::LazyLock;

use serde::Deserialize;

use crate::orchestrator::components::route::{
    routes_for_mode, OrchestratorMode, Route, RouteKind, RouteSlots,
};
use crate::orchestrator::components::tool_call::SpeedPreset;
use crate::orchestrator::systems::normalize::normalize_utterance;

const RULE_TABLE_JSON: &str = include_str!("../data/keyword_router_rules.json");

#[derive(Deserialize)]
struct SpeedModifierEntry {
    preset: String,
    terms: Vec<String>,
}

#[derive(Deserialize)]
struct RawRule {
    route: String,
    #[serde(default)]
    requires_interrogative: bool,
    #[serde(default)]
    required: Vec<Vec<String>>,
    #[serde(default)]
    forbidden: Vec<String>,
}

#[derive(Deserialize)]
struct RawRuleTable {
    interrogative_trailing: Vec<String>,
    interrogative_contains: Vec<String>,
    speed_modifiers: Vec<SpeedModifierEntry>,
    rules: Vec<RawRule>,
}

struct KeywordRule {
    route: Route,
    requires_interrogative: bool,
    required: Vec<Vec<String>>,
    forbidden: Vec<String>,
}

struct RuleTable {
    interrogative_trailing: Vec<String>,
    interrogative_contains: Vec<String>,
    speed_modifiers: Vec<(SpeedPreset, Vec<String>)>,
    rules: Vec<KeywordRule>,
}

static RULE_TABLE: LazyLock<RuleTable> = LazyLock::new(load_rule_table);

fn load_rule_table() -> RuleTable {
    let raw: RawRuleTable = serde_json::from_str(RULE_TABLE_JSON)
        .expect("keyword_router_rules.json is embedded at compile time and must parse");

    let speed_modifiers = raw
        .speed_modifiers
        .into_iter()
        .map(|entry| {
            let preset = SpeedPreset::from_str(&entry.preset)
                .unwrap_or_else(|| panic!("unknown speed preset {}", entry.preset));
            (preset, entry.terms)
        })
        .collect();

    let rules = raw
        .rules
        .into_iter()
        .map(|rule| {
            let route = Route::from_id(&rule.route)
                .unwrap_or_else(|| panic!("unknown route id {}", rule.route));
            KeywordRule {
                route,
                requires_interrogative: rule.requires_interrogative,
                required: rule.required,
                forbidden: rule.forbidden,
            }
        })
        .collect();

    RuleTable {
        interrogative_trailing: raw.interrogative_trailing,
        interrogative_contains: raw.interrogative_contains,
        speed_modifiers,
        rules,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoutingDecision {
    /// A keyword rule matched. The embedding router is skipped entirely.
    Decided(Route),
    /// No rule matched. The embedding router ranks these candidates and nothing else.
    Narrowed(Vec<Route>),
}

#[derive(Clone, Debug)]
pub struct KeywordRoutingOutcome {
    pub normalized: String,
    pub slots: RouteSlots,
    pub decision: RoutingDecision,
}

pub fn route_by_keyword(utterance: &str, mode: OrchestratorMode) -> KeywordRoutingOutcome {
    let normalized = normalize_utterance(utterance);
    let candidates = narrow_candidates(&normalized, mode);
    let slots = RouteSlots {
        object_name: None,
        speed: extract_speed_modifier(&normalized),
    };

    let interrogative = is_interrogative(&normalized);
    let decision = match find_matching_rule(&normalized, &candidates, interrogative) {
        Some(route) => RoutingDecision::Decided(route),
        None => RoutingDecision::Narrowed(candidates),
    };

    KeywordRoutingOutcome {
        normalized,
        slots,
        decision,
    }
}

/// A question must not be able to reach an edit route. This is the same
/// structural defence as Read Only mode, applied per utterance.
pub fn is_interrogative(normalized: &str) -> bool {
    let table = &*RULE_TABLE;
    let trailing = table
        .interrogative_trailing
        .iter()
        .any(|marker| normalized.ends_with(marker.as_str()));

    trailing
        || table
            .interrogative_contains
            .iter()
            .any(|marker| normalized.contains(marker.as_str()))
}

pub fn extract_speed_modifier(normalized: &str) -> Option<SpeedPreset> {
    RULE_TABLE
        .speed_modifiers
        .iter()
        .find(|(_, terms)| terms.iter().any(|term| normalized.contains(term.as_str())))
        .map(|(preset, _)| *preset)
}

fn narrow_candidates(normalized: &str, mode: OrchestratorMode) -> Vec<Route> {
    let candidates = routes_for_mode(mode);
    if !is_interrogative(normalized) {
        return candidates;
    }

    candidates
        .into_iter()
        .filter(|route| route.kind() == RouteKind::ReadOnly)
        .collect()
}

fn find_matching_rule(
    normalized: &str,
    candidates: &[Route],
    interrogative: bool,
) -> Option<Route> {
    RULE_TABLE
        .rules
        .iter()
        .find(|rule| {
            candidates.contains(&rule.route) && rule_matches(rule, normalized, interrogative)
        })
        .map(|rule| rule.route)
}

fn rule_matches(rule: &KeywordRule, normalized: &str, interrogative: bool) -> bool {
    if rule.requires_interrogative && !interrogative {
        return false;
    }

    let forbidden_hit = rule
        .forbidden
        .iter()
        .any(|term| normalized.contains(term.as_str()));
    if forbidden_hit {
        return false;
    }

    rule.required
        .iter()
        .all(|group| group.iter().any(|term| normalized.contains(term.as_str())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::components::route::ALL_ROUTES;

    fn decide(utterance: &str) -> Option<Route> {
        match route_by_keyword(utterance, OrchestratorMode::AllowEdit).decision {
            RoutingDecision::Decided(route) => Some(route),
            RoutingDecision::Narrowed(_) => None,
        }
    }

    fn candidates(utterance: &str) -> Vec<Route> {
        match route_by_keyword(utterance, OrchestratorMode::AllowEdit).decision {
            RoutingDecision::Decided(route) => vec![route],
            RoutingDecision::Narrowed(routes) => routes,
        }
    }

    fn assert_decides(utterance: &str, expected_id: &str) {
        let decided = decide(utterance).unwrap_or_else(|| {
            panic!("{utterance:?} fell through to the embedding router instead of deciding")
        });
        assert_eq!(decided.id(), expected_id, "for {utterance:?}");
    }

    #[test]
    fn the_rule_table_parses_and_every_route_id_resolves() {
        assert!(!RULE_TABLE.rules.is_empty());
        assert_eq!(RULE_TABLE.speed_modifiers.len(), 3);
    }

    #[test]
    fn every_route_referenced_by_a_rule_exists_in_the_route_table() {
        for rule in &RULE_TABLE.rules {
            assert!(ALL_ROUTES.contains(&rule.route), "{}", rule.route.id());
        }
    }

    #[test]
    fn rule_terms_are_lowercase_so_they_match_normalized_text() {
        for rule in &RULE_TABLE.rules {
            for term in rule.required.iter().flatten().chain(rule.forbidden.iter()) {
                assert_eq!(term.as_str(), term.to_lowercase(), "in {}", rule.route.id());
            }
        }
    }

    #[test]
    fn seek_start_and_end_are_separated() {
        assert_decides("jump to the start of the timeline", "seek_time:start");
        assert_decides("go to the end of the timeline", "seek_time:end");
        assert_decides("タイムラインの最後に移動して", "seek_time:end");
        assert_decides("先頭に移動して", "seek_time:start");
    }

    #[test]
    fn a_stop_that_mentions_the_start_is_not_a_seek() {
        assert_decides(
            "stop the animation and go back to the start",
            "stop_animation",
        );
        assert_decides("再生を止めて最初に戻して", "stop_animation");
    }

    #[test]
    fn keyframe_stepping_is_separated_by_direction() {
        assert_decides("move to the next keyframe", "seek_time:next_key");
        assert_decides("go back one keyframe", "seek_time:prev_key");
    }

    #[test]
    fn undo_and_redo_are_separated() {
        assert_decides("undo that", "undo");
        assert_decides("revert my last change", "undo");
        assert_decides("今の操作を取り消して", "undo");
        assert_decides("redo it", "redo");
        assert_decides("取り消したのを戻して", "redo");
    }

    #[test]
    fn pause_and_stop_are_separated() {
        assert_decides("pause the animation", "pause_animation");
        assert_decides("stop playback", "stop_animation");
        assert_decides("再生を一時停止して", "pause_animation");
        assert_decides("halt the animation and rewind it", "stop_animation");
    }

    #[test]
    fn looping_beats_both_pause_and_stop() {
        assert_decides("stop looping the animation", "toggle_loop");
        assert_decides("turn looping on", "toggle_loop");
        assert_decides("ループ再生を切り替えて", "toggle_loop");
    }

    #[test]
    fn show_and_hide_are_separated() {
        assert_decides("hide the cube", "set_object_visibility:hide");
        assert_decides("show the floor again", "set_object_visibility:show");
        assert_decides("make hero invisible", "set_object_visibility:hide");
        assert_decides("床を非表示にして", "set_object_visibility:hide");
    }

    #[test]
    fn listing_objects_is_not_a_visibility_change() {
        assert_decides("show me the object names", "list_objects");
    }

    #[test]
    fn play_is_not_confused_with_speed_or_motion_generation() {
        assert_decides("play the animation", "play_animation");
        assert_decides("start playback", "play_animation");
        assert_decides("run the animation now", "play_animation");
        assert_decides("アニメーションを再生して", "play_animation");
    }

    #[test]
    fn a_speed_change_still_reaches_the_speed_route() {
        assert_decides("play it slower", "set_playback_speed:slow");
        assert_decides("make the playback fast", "set_playback_speed:fast");
        assert_decides(
            "reset the playback speed to normal",
            "set_playback_speed:normal",
        );
        assert_decides("もっとゆっくり再生して", "set_playback_speed:slow");
    }

    #[test]
    fn motion_generation_beats_the_speed_rules() {
        assert_decides("generate a slow walk", "generate_motion:walk");
        assert_decides("make him run fast", "generate_motion:run");
        assert_decides("create an idle animation", "generate_motion:idle");
        assert_decides("ゆっくり歩くモーションを作って", "generate_motion:walk");
    }

    #[test]
    fn selection_and_camera_focus_are_separated() {
        assert_decides("select the cube", "select_object");
        assert_decides("キューブを選択して", "select_object");
        assert_decides(
            "focus the camera on the selection",
            "focus_camera:selection",
        );
        assert_decides("カメラを選択物に合わせて", "focus_camera:selection");
        assert_decides("frame the whole model", "focus_camera:model");
        assert_decides("reset the camera", "focus_camera:reset");
    }

    #[test]
    fn a_camera_reset_is_not_a_speed_reset() {
        assert_decides(
            "reset the playback speed to normal",
            "set_playback_speed:normal",
        );
    }

    #[test]
    fn a_question_can_never_reach_an_edit_route() {
        for utterance in [
            "is the animation playing?",
            "what is the current playback speed?",
            "what is selected right now?",
            "再生中かどうか教えて",
            "今何が選択されている？",
        ] {
            for route in candidates(utterance) {
                assert_eq!(
                    route.kind(),
                    RouteKind::ReadOnly,
                    "{utterance:?} kept edit route {}",
                    route.id()
                );
            }
        }
    }

    #[test]
    fn questions_about_playback_resolve_to_the_state_query() {
        assert_decides("is the animation playing?", "get_playback_state");
        assert_decides("what is the current playback speed?", "get_playback_state");
        assert_decides("再生中かどうか教えて", "get_playback_state");
    }

    #[test]
    fn questions_about_the_selection_resolve_to_the_selection_query() {
        assert_decides("what is selected right now?", "describe_selection");
        assert_decides("describe what I have selected", "describe_selection");
        assert_decides("今何が選択されている？", "describe_selection");
    }

    #[test]
    fn screenshots_are_recognised_without_a_question() {
        assert_decides("take a screenshot", "take_screenshot");
        assert_decides("capture the viewport", "take_screenshot");
        assert_decides("grab an image of the current view", "take_screenshot");
        assert_decides("スクリーンショットを撮って", "take_screenshot");
    }

    #[test]
    fn saving_is_not_confused_with_screenshots() {
        assert_decides("save the scene", "save_scene");
        assert_decides("シーンを保存して", "save_scene");
        assert_eq!(
            decide("save a picture of the viewport").map(|r| r.id()),
            Some("take_screenshot".to_string())
        );
    }

    #[test]
    fn unsupported_and_vague_requests_are_never_decided() {
        for utterance in [
            "change it",
            "make it better",
            "do the thing we talked about",
            "いい感じにして",
            "export this to FBX",
            "add a cloth physics simulation to the cape",
            "render this in 4K with ray tracing",
            "FBXで書き出して",
        ] {
            assert_eq!(decide(utterance), None, "{utterance:?} was wrongly decided");
        }
    }

    #[test]
    fn speed_modifiers_are_extracted_without_deciding_a_route() {
        let output = route_by_keyword(
            "ゆっくり歩くモーションを作って",
            OrchestratorMode::AllowEdit,
        );
        assert_eq!(output.slots.speed, Some(SpeedPreset::Slow));

        let output = route_by_keyword("make him run fast", OrchestratorMode::AllowEdit);
        assert_eq!(output.slots.speed, Some(SpeedPreset::Fast));
    }

    #[test]
    fn read_only_mode_still_cannot_decide_an_edit_route() {
        let output = route_by_keyword("hide the cube", OrchestratorMode::ReadOnly);
        match output.decision {
            RoutingDecision::Decided(route) => {
                assert_eq!(route.kind(), RouteKind::ReadOnly)
            }
            RoutingDecision::Narrowed(routes) => {
                assert!(routes
                    .iter()
                    .all(|route| route.kind() == RouteKind::ReadOnly))
            }
        }
    }
}
