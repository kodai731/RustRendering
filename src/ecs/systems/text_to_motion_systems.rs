use crate::ecs::resource::{TextToMotionState, TextToMotionStatus};
use crate::grpc::{GrpcRequest, GrpcThreadHandle, TextToMotionRequest};

pub fn text_to_motion_submit(
    state: &mut TextToMotionState,
    handle: &GrpcThreadHandle,
    prompt: &str,
    duration_seconds: f32,
) {
    state.status = TextToMotionStatus::Generating;
    state.last_prompt = prompt.to_string();
    state.last_duration = duration_seconds;
    state.error_message = None;
    state.generated_clip = None;
    state.generation_time_ms = None;
    state.model_used = None;

    let request = TextToMotionRequest {
        prompt: prompt.to_string(),
        duration_seconds,
        target_fps: 30,
    };

    handle.send(GrpcRequest::GenerateMotion(request));
    log!(
        "TextToMotion: submitted '{}' (duration={}s)",
        prompt,
        duration_seconds
    );
}

pub fn text_to_motion_cancel(state: &mut TextToMotionState) {
    state.status = TextToMotionStatus::Idle;
    state.generated_clip = None;
    state.error_message = None;
    state.generation_time_ms = None;
    state.model_used = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_to_motion_cancel() {
        let mut state = TextToMotionState::default();
        state.status = TextToMotionStatus::Generating;
        state.error_message = Some("test error".to_string());

        text_to_motion_cancel(&mut state);

        assert_eq!(state.status, TextToMotionStatus::Idle);
        assert!(state.generated_clip.is_none());
        assert!(state.error_message.is_none());
    }
}
