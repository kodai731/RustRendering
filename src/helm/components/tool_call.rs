use crate::ecs::world::Visibility;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeedPreset {
    Slow,
    Normal,
    Fast,
}

impl SpeedPreset {
    pub fn to_multiplier(self) -> f32 {
        match self {
            SpeedPreset::Slow => 0.5,
            SpeedPreset::Normal => 1.0,
            SpeedPreset::Fast => 2.0,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SpeedPreset::Slow => "slow",
            SpeedPreset::Normal => "normal",
            SpeedPreset::Fast => "fast",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeekPosition {
    Start,
    End,
    NextKey,
    PrevKey,
}

impl SeekPosition {
    pub fn as_str(self) -> &'static str {
        match self {
            SeekPosition::Start => "start",
            SeekPosition::End => "end",
            SeekPosition::NextKey => "next_key",
            SeekPosition::PrevKey => "prev_key",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisibilityState {
    Show,
    Hide,
}

impl VisibilityState {
    pub fn to_visibility(self) -> Visibility {
        match self {
            VisibilityState::Show => Visibility::Shown,
            VisibilityState::Hide => Visibility::Hidden,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            VisibilityState::Show => "show",
            VisibilityState::Hide => "hide",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FocusTarget {
    Selection,
    Model,
    Reset,
}

impl FocusTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            FocusTarget::Selection => "selection",
            FocusTarget::Model => "model",
            FocusTarget::Reset => "reset",
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ShotPreset {
    LookAtSelection,
    OrbitAroundSelection,
    DollyIn,
    DollyOut,
    CraneUp,
    CraneDown,
}

impl ShotPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            ShotPreset::LookAtSelection => "look_at_selection",
            ShotPreset::OrbitAroundSelection => "orbit_around_selection",
            ShotPreset::DollyIn => "dolly_in",
            ShotPreset::DollyOut => "dolly_out",
            ShotPreset::CraneUp => "crane_up",
            ShotPreset::CraneDown => "crane_down",
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum MotionCategory {
    Walk,
    Run,
    Idle,
    Jump,
    Turn,
}

impl MotionCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            MotionCategory::Walk => "walk",
            MotionCategory::Run => "run",
            MotionCategory::Idle => "idle",
            MotionCategory::Jump => "jump",
            MotionCategory::Turn => "turn",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectName(pub String);

impl ObjectName {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A fully bound tool invocation. Every argument is a resolved enum or an owned
/// name, so a value of this type cannot describe an unsupported operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolCall {
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
    SelectObject(ObjectName),
    SetObjectVisibility(ObjectName, VisibilityState),
    FocusCamera(FocusTarget),
    Undo,
    Redo,
    SaveScene,
    GenerateMotion(MotionCategory, SpeedPreset),
    CameraShot(ShotPreset, SpeedPreset),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiskLevel {
    ReadOnly,
    Mutating,
    Destructive,
}

impl RiskLevel {
    pub fn requires_confirmation(self) -> bool {
        self == RiskLevel::Destructive
    }
}

impl ToolCall {
    pub fn risk_level(&self) -> RiskLevel {
        match self {
            ToolCall::ListObjects
            | ToolCall::DescribeSelection
            | ToolCall::GetPlaybackState
            | ToolCall::TakeScreenshot => RiskLevel::ReadOnly,

            ToolCall::PlayAnimation
            | ToolCall::PauseAnimation
            | ToolCall::StopAnimation
            | ToolCall::SetPlaybackSpeed(_)
            | ToolCall::SeekTime(_)
            | ToolCall::ToggleLoop
            | ToolCall::SelectObject(_)
            | ToolCall::SetObjectVisibility(_, _)
            | ToolCall::FocusCamera(_)
            | ToolCall::CameraShot(_, _)
            | ToolCall::Undo
            | ToolCall::Redo => RiskLevel::Mutating,

            ToolCall::SaveScene | ToolCall::GenerateMotion(_, _) => RiskLevel::Destructive,
        }
    }

    pub fn tool_name(&self) -> &'static str {
        match self {
            ToolCall::ListObjects => "list_objects",
            ToolCall::DescribeSelection => "describe_selection",
            ToolCall::GetPlaybackState => "get_playback_state",
            ToolCall::TakeScreenshot => "take_screenshot",
            ToolCall::PlayAnimation => "play_animation",
            ToolCall::PauseAnimation => "pause_animation",
            ToolCall::StopAnimation => "stop_animation",
            ToolCall::SetPlaybackSpeed(_) => "set_playback_speed",
            ToolCall::SeekTime(_) => "seek_time",
            ToolCall::ToggleLoop => "toggle_loop",
            ToolCall::SelectObject(_) => "select_object",
            ToolCall::SetObjectVisibility(_, _) => "set_object_visibility",
            ToolCall::FocusCamera(_) => "focus_camera",
            ToolCall::Undo => "undo",
            ToolCall::Redo => "redo",
            ToolCall::SaveScene => "save_scene",
            ToolCall::GenerateMotion(_, _) => "generate_motion",
            ToolCall::CameraShot(_, _) => "camera_shot",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_presets_map_to_distinct_multipliers() {
        assert_eq!(SpeedPreset::Slow.to_multiplier(), 0.5);
        assert_eq!(SpeedPreset::Normal.to_multiplier(), 1.0);
        assert_eq!(SpeedPreset::Fast.to_multiplier(), 2.0);
    }

    #[test]
    fn visibility_state_maps_to_engine_visibility() {
        assert_eq!(VisibilityState::Show.to_visibility(), Visibility::Shown);
        assert_eq!(VisibilityState::Hide.to_visibility(), Visibility::Hidden);
    }

    #[test]
    fn query_tools_are_read_only() {
        assert_eq!(ToolCall::ListObjects.risk_level(), RiskLevel::ReadOnly);
        assert_eq!(
            ToolCall::DescribeSelection.risk_level(),
            RiskLevel::ReadOnly
        );
        assert_eq!(ToolCall::GetPlaybackState.risk_level(), RiskLevel::ReadOnly);
        assert_eq!(ToolCall::TakeScreenshot.risk_level(), RiskLevel::ReadOnly);
    }

    #[test]
    fn destructive_tools_require_confirmation() {
        assert!(ToolCall::SaveScene.risk_level().requires_confirmation());
        assert!(
            ToolCall::GenerateMotion(MotionCategory::Walk, SpeedPreset::Normal)
                .risk_level()
                .requires_confirmation()
        );
        assert!(!ToolCall::PlayAnimation.risk_level().requires_confirmation());
    }
}
