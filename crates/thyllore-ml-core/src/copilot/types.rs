pub type InferenceActorId = u64;
pub type InferenceRequestId = u64;

pub const CURVE_COPILOT_ACTOR_ID: InferenceActorId = 2;

#[derive(Clone, Debug)]
pub enum InferenceModelKind {
    CurveCopilot,
}

#[derive(Clone, Debug)]
pub enum InferenceRequestKind {
    CurveCopilotPredict {
        context: Vec<f32>,
        future: Vec<f32>,
        reveal_mask: Vec<bool>,
        fps: f32,
    },
}

#[derive(Clone, Debug)]
pub struct InferenceRequest {
    pub request_id: InferenceRequestId,
    pub actor_id: InferenceActorId,
    pub kind: InferenceRequestKind,
}

#[derive(Clone, Debug)]
pub enum InferenceResultKind {
    CurveCopilotPredict { mean_curve: Vec<f32> },
}

#[derive(Clone, Debug)]
pub struct InferenceResult {
    pub request_id: InferenceRequestId,
    pub actor_id: InferenceActorId,
    pub kind: InferenceResultKind,
}
