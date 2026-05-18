use anyhow::Result;
use ort::session::Session as OrtSession;

use super::input::{build_curve_copilot_tensors, build_curve_predict_tensor, BoneContextInput};
use super::output::{parse_curve_copilot_output, parse_curve_predict_output};
use super::types::CopilotStepPrediction;

const MAX_STEPS_METADATA_KEY: &str = "max_steps";

pub struct Session {
    ort: OrtSession,
    max_steps: Option<usize>,
}

pub struct CurveCopilotRequest<'a> {
    pub context: &'a [f32],
    pub property_type_id: u32,
    pub topology_features: &'a [f32],
    pub bone_name_tokens: &'a [i64],
    pub query_times: &'a [f32],
    pub curve_window: &'a [f32],
    pub bone_context: BoneContextInput<'a>,
}

impl Session {
    pub fn from_onnx_path(path: &str) -> Result<Self> {
        let ort = OrtSession::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(path)?;
        let max_steps = read_max_steps_metadata(&ort);
        Ok(Self { ort, max_steps })
    }

    pub fn max_steps(&self) -> Option<usize> {
        self.max_steps
    }

    pub fn run_curve_predict(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let tensor = build_curve_predict_tensor(input)?;
        let outputs = self.ort.run(ort::inputs![tensor])?;
        parse_curve_predict_output(&outputs)
    }

    pub fn run_curve_copilot(
        &mut self,
        request: CurveCopilotRequest<'_>,
    ) -> Result<Vec<CopilotStepPrediction>> {
        let tensors = build_curve_copilot_tensors(
            request.context,
            request.property_type_id,
            request.topology_features,
            request.bone_name_tokens,
            request.query_times,
            request.curve_window,
            request.bone_context,
        )?;

        let outputs = self.ort.run(ort::inputs![
            "context_keyframes" => tensors.context,
            "property_type" => tensors.property_type,
            "topology_features" => tensors.topology,
            "bone_name_tokens" => tensors.bone_name,
            "query_times" => tensors.query_times,
            "bone_context_keyframes" => tensors.bone_context_keyframes,
            "bone_topology_features" => tensors.bone_context_topology,
            "bone_rest_positions" => tensors.bone_context_rest_positions,
            "bone_context_mask" => tensors.bone_context_mask,
            "curve_window" => tensors.curve_window
        ])?;

        parse_curve_copilot_output(&outputs, request.query_times.len())
    }
}

fn read_max_steps_metadata(ort: &OrtSession) -> Option<usize> {
    let metadata = ort.metadata().ok()?;
    let value = metadata.custom(MAX_STEPS_METADATA_KEY)?;
    value.parse::<usize>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_onnx_path_returns_error_for_missing_file() {
        let result = Session::from_onnx_path("/nonexistent/path/model.onnx");
        assert!(result.is_err());
    }
}
