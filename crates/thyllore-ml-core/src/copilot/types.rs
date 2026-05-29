pub type InferenceActorId = u64;
pub type InferenceRequestId = u64;

pub const CURVE_COPILOT_ACTOR_ID: InferenceActorId = 2;

#[derive(Clone, Debug)]
pub enum InferenceModelKind {
    CurvePredictor,
    CurveCopilot,
}

#[derive(Clone, Debug)]
pub enum InferenceRequestKind {
    CurvePredict {
        input: Vec<f32>,
    },
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
pub struct CopilotStepPrediction {
    pub value: f32,
    pub tangent_in: (f32, f32),
    pub tangent_out: (f32, f32),
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub enum InferenceResultKind {
    CurvePredict { output: Vec<f32> },
    CurveCopilotPredict { mean_curve: Vec<f32> },
}

#[derive(Clone, Debug)]
pub struct InferenceResult {
    pub request_id: InferenceRequestId,
    pub actor_id: InferenceActorId,
    pub kind: InferenceResultKind,
}
