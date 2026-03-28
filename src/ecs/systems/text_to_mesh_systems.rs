use crate::ecs::resource::{PendingMeshRequest, TextToMeshState, TextToMeshStatus};
use crate::grpc::{GrpcRequest, GrpcThreadHandle, MeshInputMode, TextToMeshRequest};

pub fn text_to_mesh_submit(
    state: &mut TextToMeshState,
    handle: &GrpcThreadHandle,
    prompt: String,
    target_faces: u32,
    seed: u32,
    input_mode: MeshInputMode,
    input_image_png: Option<Vec<u8>>,
) {
    state.status = TextToMeshStatus::WaitingForServer;
    state.last_prompt = prompt.clone();
    state.glb_data = None;
    state.error_message = None;
    state.generation_time_ms = None;
    state.vertex_count = None;
    state.face_count = None;
    state.intermediate_image_png = None;
    state.last_status_check = None;

    state.pending_request = Some(PendingMeshRequest {
        prompt,
        target_faces,
        seed,
        input_mode,
        input_image_png,
    });

    handle.send(GrpcRequest::CheckMeshStatus);
    log!("TextToMesh: waiting for server readiness...");
}

pub fn text_to_mesh_send_generate(state: &mut TextToMeshState, handle: &GrpcThreadHandle) {
    let pending = match state.pending_request.take() {
        Some(p) => p,
        None => return,
    };

    state.status = TextToMeshStatus::Generating;

    let has_image = pending.input_image_png.is_some();
    handle.send(GrpcRequest::GenerateMesh(TextToMeshRequest {
        prompt: pending.prompt.clone(),
        target_faces: pending.target_faces,
        seed: pending.seed,
        input_mode: pending.input_mode.clone(),
        input_image_png: pending.input_image_png,
    }));

    log!(
        "TextToMesh: submitted '{}' (mode={:?}, faces={}, seed={}, has_image={})",
        pending.prompt,
        pending.input_mode,
        pending.target_faces,
        pending.seed,
        has_image
    );
}

pub fn text_to_mesh_cancel(state: &mut TextToMeshState) {
    state.status = TextToMeshStatus::Idle;
    state.glb_data = None;
    state.error_message = None;
    state.generation_time_ms = None;
    state.vertex_count = None;
    state.face_count = None;
    state.intermediate_image_png = None;
    state.pending_request = None;
    state.last_status_check = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_to_mesh_cancel() {
        let mut state = TextToMeshState::default();
        state.status = TextToMeshStatus::Generating;
        state.error_message = Some("test error".to_string());

        text_to_mesh_cancel(&mut state);

        assert_eq!(state.status, TextToMeshStatus::Idle);
        assert!(state.glb_data.is_none());
        assert!(state.error_message.is_none());
        assert!(state.pending_request.is_none());
    }
}
