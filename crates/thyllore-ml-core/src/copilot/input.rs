use anyhow::Result;
use ort::value::Tensor;

pub const CONTEXT_KEYFRAME_COUNT: i64 = 8;
pub const CONTEXT_FEATURE_DIM: i64 = 6;
pub const TOPOLOGY_FEATURE_DIM: i64 = 6;
pub const BONE_NAME_TOKEN_DIM: i64 = 32;

pub struct CurveCopilotTensors {
    pub context: Tensor<f32>,
    pub property_type: Tensor<i64>,
    pub topology: Tensor<f32>,
    pub bone_name: Tensor<i64>,
    pub query_times: Tensor<f32>,
    pub curve_window: Tensor<f32>,
}

pub fn build_curve_predict_tensor(input: &[f32]) -> Result<Tensor<f32>> {
    let input_len = input.len() as i64;
    let tensor = Tensor::from_array((vec![1i64, input_len], input.to_vec()))?;
    Ok(tensor)
}

pub fn build_curve_copilot_tensors(
    context: &[f32],
    property_type_id: u32,
    topology_features: &[f32],
    bone_name_tokens: &[i64],
    query_times: &[f32],
    curve_window: &[f32],
) -> Result<CurveCopilotTensors> {
    let num_steps = query_times.len() as i64;
    let curve_window_len = curve_window.len() as i64;

    let context = Tensor::from_array((
        vec![1i64, CONTEXT_KEYFRAME_COUNT, CONTEXT_FEATURE_DIM],
        context.to_vec(),
    ))?;
    let property_type = Tensor::from_array((vec![1i64], vec![property_type_id as i64]))?;
    let topology =
        Tensor::from_array((vec![1i64, TOPOLOGY_FEATURE_DIM], topology_features.to_vec()))?;
    let bone_name =
        Tensor::from_array((vec![1i64, BONE_NAME_TOKEN_DIM], bone_name_tokens.to_vec()))?;
    let query_times = Tensor::from_array((vec![1i64, num_steps], query_times.to_vec()))?;
    let curve_window = Tensor::from_array((vec![1i64, curve_window_len], curve_window.to_vec()))?;

    Ok(CurveCopilotTensors {
        context,
        property_type,
        topology,
        bone_name,
        query_times,
        curve_window,
    })
}
