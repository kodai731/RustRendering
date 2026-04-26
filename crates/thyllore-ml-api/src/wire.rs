use serde::{Deserialize, Serialize};

use crate::request::CopilotRequest;
use crate::response::{CopilotResponse, CopilotStepPrediction};

pub const OP_RUN_CURVE_COPILOT: &str = "run_curve_copilot";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopilotRequestWire {
    pub model_path: String,
    pub property_type: u32,
    pub context_bits: Vec<u32>,
    pub topology_features_bits: Vec<u32>,
    pub bone_name_tokens: Vec<i64>,
    pub query_times_bits: Vec<u32>,
    pub curve_window_bits: Vec<u32>,
}

impl CopilotRequestWire {
    pub fn from_typed(request: &CopilotRequest) -> Self {
        Self {
            model_path: request.model_path.clone(),
            property_type: request.property_type,
            context_bits: request.context.iter().map(|v| v.to_bits()).collect(),
            topology_features_bits: request
                .topology_features
                .iter()
                .map(|v| v.to_bits())
                .collect(),
            bone_name_tokens: request.bone_name_tokens.clone(),
            query_times_bits: request.query_times.iter().map(|v| v.to_bits()).collect(),
            curve_window_bits: request.curve_window.iter().map(|v| v.to_bits()).collect(),
        }
    }

    pub fn into_typed(self) -> CopilotRequest {
        CopilotRequest {
            model_path: self.model_path,
            property_type: self.property_type,
            context: self.context_bits.into_iter().map(f32::from_bits).collect(),
            topology_features: self
                .topology_features_bits
                .into_iter()
                .map(f32::from_bits)
                .collect(),
            bone_name_tokens: self.bone_name_tokens,
            query_times: self
                .query_times_bits
                .into_iter()
                .map(f32::from_bits)
                .collect(),
            curve_window: self
                .curve_window_bits
                .into_iter()
                .map(f32::from_bits)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopilotStepWire {
    pub value_bits: u32,
    pub tangent_in_bits: [u32; 2],
    pub tangent_out_bits: [u32; 2],
    pub confidence_bits: u32,
}

impl CopilotStepWire {
    pub fn from_typed(step: &CopilotStepPrediction) -> Self {
        Self {
            value_bits: step.value.to_bits(),
            tangent_in_bits: [step.tangent_in[0].to_bits(), step.tangent_in[1].to_bits()],
            tangent_out_bits: [step.tangent_out[0].to_bits(), step.tangent_out[1].to_bits()],
            confidence_bits: step.confidence.to_bits(),
        }
    }

    pub fn into_typed(self) -> CopilotStepPrediction {
        CopilotStepPrediction {
            value: f32::from_bits(self.value_bits),
            tangent_in: [
                f32::from_bits(self.tangent_in_bits[0]),
                f32::from_bits(self.tangent_in_bits[1]),
            ],
            tangent_out: [
                f32::from_bits(self.tangent_out_bits[0]),
                f32::from_bits(self.tangent_out_bits[1]),
            ],
            confidence: f32::from_bits(self.confidence_bits),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopilotResponseWire {
    pub predictions: Vec<CopilotStepWire>,
}

impl CopilotResponseWire {
    pub fn from_typed(response: &CopilotResponse) -> Self {
        Self {
            predictions: response
                .predictions
                .iter()
                .map(CopilotStepWire::from_typed)
                .collect(),
        }
    }

    pub fn into_typed(self) -> CopilotResponse {
        CopilotResponse {
            predictions: self
                .predictions
                .into_iter()
                .map(CopilotStepWire::into_typed)
                .collect(),
        }
    }
}
