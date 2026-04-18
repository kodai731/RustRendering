use crate::ecs::resource::{AutoRigState, AutoRigStatus};
use crate::ecs::world::Entity;
use crate::grpc::{GrpcRequest, GrpcThreadHandle};

pub fn auto_rig_submit(
    state: &mut AutoRigState,
    handle: &GrpcThreadHandle,
    source_glb_data: Vec<u8>,
    target_entity: Entity,
) {
    state.status = AutoRigStatus::WaitingForServer;
    state.rigged_glb_data = None;
    state.error_message = None;
    state.joint_count = None;
    state.bone_count = None;
    state.generation_time_ms = None;
    state.source_glb_data = Some(source_glb_data);
    state.target_entity = Some(target_entity);
    state.last_status_check = None;

    handle.send(GrpcRequest::CheckRiggingStatus);
}

pub fn auto_rig_send_generate(state: &mut AutoRigState, handle: &GrpcThreadHandle) {
    let glb_data = match state.source_glb_data.take() {
        Some(data) => data,
        None => return,
    };

    state.status = AutoRigStatus::Rigging;

    handle.send(GrpcRequest::GenerateRig(crate::grpc::RiggingRequest {
        glb_data,
        num_sample_points: 65536,
    }));
}

pub fn auto_rig_cancel(state: &mut AutoRigState) {
    state.status = AutoRigStatus::Idle;
    state.rigged_glb_data = None;
    state.error_message = None;
    state.source_glb_data = None;
    state.original_glb_backup = None;
    state.target_entity = None;
    state.last_status_check = None;
}
