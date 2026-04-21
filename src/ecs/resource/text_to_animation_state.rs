use crate::animation::editable::EditableAnimationClip;
use crate::ecs::world::Entity;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TextToAnimationStatus {
    Idle,
    AutoRigging,
    AutoRigApplying,
    GeneratingMotion,
    ApplyingClip,
    Done,
    Error,
}

pub struct TextToAnimationState {
    pub status: TextToAnimationStatus,
    pub target_entity: Option<Entity>,
    pub prompt: String,
    pub duration_seconds: f32,
    pub error_message: Option<String>,
    pub generation_time_ms: Option<f32>,
    pub model_used: Option<String>,
    pub generated_clip: Option<EditableAnimationClip>,
    pub skipped_auto_rig: bool,
    pub server_ready: bool,
}

impl Default for TextToAnimationState {
    fn default() -> Self {
        Self {
            status: TextToAnimationStatus::Idle,
            target_entity: None,
            prompt: String::new(),
            duration_seconds: 3.0,
            error_message: None,
            generation_time_ms: None,
            model_used: None,
            generated_clip: None,
            skipped_auto_rig: false,
            server_ready: false,
        }
    }
}
