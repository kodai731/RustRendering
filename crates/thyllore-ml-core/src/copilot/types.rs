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
        property_type_id: u32,
        topology_features: Vec<f32>,
        bone_name_tokens: Vec<i64>,
        query_times: Vec<f32>,
        curve_window: Vec<f32>,
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
    CurveCopilotPredict { steps: Vec<CopilotStepPrediction> },
}

#[derive(Clone, Debug)]
pub struct InferenceResult {
    pub request_id: InferenceRequestId,
    pub actor_id: InferenceActorId,
    pub kind: InferenceResultKind,
}
