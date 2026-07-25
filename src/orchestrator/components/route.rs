//! Routes are what the router indexes: a tool combined with the enum arguments
//! that appear in an utterance. Expanding those enums into the route identity
//! leaves almost every route with no arguments left to extract, which is what
//! makes routing by similarity viable at all — a small model picks a
//! zero-argument target far more reliably than it fills one in.

use super::tool_call::{
    FocusTarget, MotionCategory, ObjectName, SeekPosition, SpeedPreset, ToolCall, VisibilityState,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteKind {
    ReadOnly,
    Edit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotKind {
    ObjectName,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrchestratorMode {
    ReadOnly,
    AllowEdit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    ListObjects,
    DescribeSelection,
    GetPlaybackState,
    TakeScreenshot,
    PlayAnimation,
    PauseAnimation,
    StopAnimation,
    SetPlaybackSpeed(SpeedPreset),
    SeekTime(SeekPosition),
    ToggleLoop,
    SelectObject,
    SetObjectVisibility(VisibilityState),
    FocusCamera(FocusTarget),
    Undo,
    Redo,
    SaveScene,
    GenerateMotion(MotionCategory),
}

pub const ALL_ROUTES: [Route; 29] = [
    Route::ListObjects,
    Route::DescribeSelection,
    Route::GetPlaybackState,
    Route::TakeScreenshot,
    Route::PlayAnimation,
    Route::PauseAnimation,
    Route::StopAnimation,
    Route::SetPlaybackSpeed(SpeedPreset::Slow),
    Route::SetPlaybackSpeed(SpeedPreset::Normal),
    Route::SetPlaybackSpeed(SpeedPreset::Fast),
    Route::SeekTime(SeekPosition::Start),
    Route::SeekTime(SeekPosition::End),
    Route::SeekTime(SeekPosition::NextKey),
    Route::SeekTime(SeekPosition::PrevKey),
    Route::ToggleLoop,
    Route::SelectObject,
    Route::SetObjectVisibility(VisibilityState::Show),
    Route::SetObjectVisibility(VisibilityState::Hide),
    Route::FocusCamera(FocusTarget::Selection),
    Route::FocusCamera(FocusTarget::Model),
    Route::FocusCamera(FocusTarget::Reset),
    Route::Undo,
    Route::Redo,
    Route::SaveScene,
    Route::GenerateMotion(MotionCategory::Walk),
    Route::GenerateMotion(MotionCategory::Run),
    Route::GenerateMotion(MotionCategory::Idle),
    Route::GenerateMotion(MotionCategory::Jump),
    Route::GenerateMotion(MotionCategory::Turn),
];

/// Values the router fills in before a route can become a `ToolCall`. Slots come
/// from name resolution, modifiers from the keyword router.
#[derive(Clone, Debug, Default)]
pub struct RouteSlots {
    pub object_name: Option<String>,
    pub speed: Option<SpeedPreset>,
}

impl RouteSlots {
    pub fn with_object_name(name: impl Into<String>) -> Self {
        Self {
            object_name: Some(name.into()),
            speed: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindError {
    MissingObjectName(Route),
}

impl Route {
    pub fn id(self) -> String {
        match self {
            Route::SetPlaybackSpeed(speed) => format!("set_playback_speed:{}", speed.as_str()),
            Route::SeekTime(position) => format!("seek_time:{}", position.as_str()),
            Route::SetObjectVisibility(state) => {
                format!("set_object_visibility:{}", state.as_str())
            }
            Route::FocusCamera(target) => format!("focus_camera:{}", target.as_str()),
            Route::GenerateMotion(category) => format!("generate_motion:{}", category.as_str()),
            other => other.tool_name().to_string(),
        }
    }

    pub fn from_id(route_id: &str) -> Option<Self> {
        ALL_ROUTES
            .iter()
            .copied()
            .find(|route| route.id() == route_id)
    }

    pub fn tool_name(self) -> &'static str {
        match self {
            Route::ListObjects => "list_objects",
            Route::DescribeSelection => "describe_selection",
            Route::GetPlaybackState => "get_playback_state",
            Route::TakeScreenshot => "take_screenshot",
            Route::PlayAnimation => "play_animation",
            Route::PauseAnimation => "pause_animation",
            Route::StopAnimation => "stop_animation",
            Route::SetPlaybackSpeed(_) => "set_playback_speed",
            Route::SeekTime(_) => "seek_time",
            Route::ToggleLoop => "toggle_loop",
            Route::SelectObject => "select_object",
            Route::SetObjectVisibility(_) => "set_object_visibility",
            Route::FocusCamera(_) => "focus_camera",
            Route::Undo => "undo",
            Route::Redo => "redo",
            Route::SaveScene => "save_scene",
            Route::GenerateMotion(_) => "generate_motion",
        }
    }

    pub fn kind(self) -> RouteKind {
        match self {
            Route::ListObjects
            | Route::DescribeSelection
            | Route::GetPlaybackState
            | Route::TakeScreenshot => RouteKind::ReadOnly,
            _ => RouteKind::Edit,
        }
    }

    pub fn slot(self) -> Option<SlotKind> {
        match self {
            Route::SelectObject | Route::SetObjectVisibility(_) => Some(SlotKind::ObjectName),
            _ => None,
        }
    }

    pub fn bind(self, slots: &RouteSlots) -> Result<ToolCall, BindError> {
        let speed = slots.speed.unwrap_or(SpeedPreset::Normal);
        match self {
            Route::ListObjects => Ok(ToolCall::ListObjects),
            Route::DescribeSelection => Ok(ToolCall::DescribeSelection),
            Route::GetPlaybackState => Ok(ToolCall::GetPlaybackState),
            Route::TakeScreenshot => Ok(ToolCall::TakeScreenshot),
            Route::PlayAnimation => Ok(ToolCall::PlayAnimation),
            Route::PauseAnimation => Ok(ToolCall::PauseAnimation),
            Route::StopAnimation => Ok(ToolCall::StopAnimation),
            Route::SetPlaybackSpeed(preset) => Ok(ToolCall::SetPlaybackSpeed(preset)),
            Route::SeekTime(position) => Ok(ToolCall::SeekTime(position)),
            Route::ToggleLoop => Ok(ToolCall::ToggleLoop),
            Route::Undo => Ok(ToolCall::Undo),
            Route::Redo => Ok(ToolCall::Redo),
            Route::SaveScene => Ok(ToolCall::SaveScene),
            Route::FocusCamera(target) => Ok(ToolCall::FocusCamera(target)),
            Route::GenerateMotion(category) => Ok(ToolCall::GenerateMotion(category, speed)),

            Route::SelectObject => Ok(ToolCall::SelectObject(self.take_object_name(slots)?)),
            Route::SetObjectVisibility(state) => Ok(ToolCall::SetObjectVisibility(
                self.take_object_name(slots)?,
                state,
            )),
        }
    }

    fn take_object_name(self, slots: &RouteSlots) -> Result<ObjectName, BindError> {
        slots
            .object_name
            .as_ref()
            .filter(|name| !name.trim().is_empty())
            .map(|name| ObjectName(name.clone()))
            .ok_or(BindError::MissingObjectName(self))
    }
}

/// Read Only mode does not reject edit routes — it never puts them in the index,
/// so the router has no way to produce one.
pub fn routes_for_mode(mode: OrchestratorMode) -> Vec<Route> {
    ALL_ROUTES
        .iter()
        .copied()
        .filter(|route| match mode {
            OrchestratorMode::ReadOnly => route.kind() == RouteKind::ReadOnly,
            OrchestratorMode::AllowEdit => true,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The router index in `scripts/orchestrator_eval/route_schema.py` must expand
    /// to exactly these ids, or the measured accuracy does not describe this build.
    const EXPECTED_ROUTE_IDS: [&str; 29] = [
        "list_objects",
        "describe_selection",
        "get_playback_state",
        "take_screenshot",
        "play_animation",
        "pause_animation",
        "stop_animation",
        "set_playback_speed:slow",
        "set_playback_speed:normal",
        "set_playback_speed:fast",
        "seek_time:start",
        "seek_time:end",
        "seek_time:next_key",
        "seek_time:prev_key",
        "toggle_loop",
        "select_object",
        "set_object_visibility:show",
        "set_object_visibility:hide",
        "focus_camera:selection",
        "focus_camera:model",
        "focus_camera:reset",
        "undo",
        "redo",
        "save_scene",
        "generate_motion:walk",
        "generate_motion:run",
        "generate_motion:idle",
        "generate_motion:jump",
        "generate_motion:turn",
    ];

    #[test]
    fn route_ids_match_the_evaluated_index() {
        let actual: Vec<String> = ALL_ROUTES.iter().map(|route| route.id()).collect();
        assert_eq!(actual, EXPECTED_ROUTE_IDS);
    }

    #[test]
    fn route_ids_are_unique() {
        let ids: HashSet<String> = ALL_ROUTES.iter().map(|route| route.id()).collect();
        assert_eq!(ids.len(), ALL_ROUTES.len());
    }

    #[test]
    fn only_object_routes_carry_a_slot() {
        let with_slot: Vec<String> = ALL_ROUTES
            .iter()
            .filter(|route| route.slot().is_some())
            .map(|route| route.id())
            .collect();
        assert_eq!(
            with_slot,
            [
                "select_object",
                "set_object_visibility:show",
                "set_object_visibility:hide"
            ]
        );
    }

    #[test]
    fn read_only_mode_excludes_every_edit_route() {
        let routes = routes_for_mode(OrchestratorMode::ReadOnly);
        assert_eq!(routes.len(), 4);
        assert!(routes
            .iter()
            .all(|route| route.kind() == RouteKind::ReadOnly));
        assert!(!routes.contains(&Route::PlayAnimation));
        assert!(!routes.contains(&Route::SaveScene));
    }

    #[test]
    fn allow_edit_mode_exposes_every_route() {
        assert_eq!(routes_for_mode(OrchestratorMode::AllowEdit).len(), 29);
    }

    #[test]
    fn every_slotless_route_binds_without_slots() {
        let slots = RouteSlots::default();
        for route in ALL_ROUTES.iter().filter(|route| route.slot().is_none()) {
            assert!(route.bind(&slots).is_ok(), "{} failed to bind", route.id());
        }
    }

    #[test]
    fn object_routes_reject_a_missing_name() {
        let slots = RouteSlots::default();
        assert_eq!(
            Route::SelectObject.bind(&slots),
            Err(BindError::MissingObjectName(Route::SelectObject))
        );
    }

    #[test]
    fn object_routes_reject_a_blank_name() {
        let slots = RouteSlots::with_object_name("   ");
        assert_eq!(
            Route::SelectObject.bind(&slots),
            Err(BindError::MissingObjectName(Route::SelectObject))
        );
    }

    #[test]
    fn object_routes_bind_the_supplied_name() {
        let slots = RouteSlots::with_object_name("Hero");
        assert_eq!(
            Route::SetObjectVisibility(VisibilityState::Hide).bind(&slots),
            Ok(ToolCall::SetObjectVisibility(
                ObjectName("Hero".to_string()),
                VisibilityState::Hide
            ))
        );
    }

    #[test]
    fn generate_motion_defaults_to_normal_speed_without_a_modifier() {
        let slots = RouteSlots::default();
        assert_eq!(
            Route::GenerateMotion(MotionCategory::Walk).bind(&slots),
            Ok(ToolCall::GenerateMotion(
                MotionCategory::Walk,
                SpeedPreset::Normal
            ))
        );
    }

    #[test]
    fn generate_motion_takes_the_speed_modifier_when_present() {
        let slots = RouteSlots {
            object_name: None,
            speed: Some(SpeedPreset::Slow),
        };
        assert_eq!(
            Route::GenerateMotion(MotionCategory::Walk).bind(&slots),
            Ok(ToolCall::GenerateMotion(
                MotionCategory::Walk,
                SpeedPreset::Slow
            ))
        );
    }
}
