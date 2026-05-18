use anyhow::Result;
use ort::value::Tensor;

pub const CONTEXT_KEYFRAME_COUNT: usize = 32;
pub const CONTEXT_FEATURE_DIM: usize = 6;
pub const TOPOLOGY_FEATURE_DIM: usize = 6;
pub const BONE_NAME_TOKEN_DIM: usize = 32;
pub const BONE_CONTEXT_N_MAX: usize = 32;
pub const BONE_CONTEXT_REST_POSITION_DIM: usize = 3;

pub struct CurveCopilotTensors {
    pub context: Tensor<f32>,
    pub property_type: Tensor<i64>,
    pub topology: Tensor<f32>,
    pub bone_name: Tensor<i64>,
    pub query_times: Tensor<f32>,
    pub curve_window: Tensor<f32>,
    pub bone_context_keyframes: Tensor<f32>,
    pub bone_context_topology: Tensor<f32>,
    pub bone_context_rest_positions: Tensor<f32>,
    pub bone_context_mask: Tensor<bool>,
}

pub struct BoneContextInput<'a> {
    pub keyframes: &'a [f32],
    pub topology: &'a [f32],
    pub rest_positions: &'a [f32],
    pub mask: &'a [bool],
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
    bone_context: BoneContextInput<'_>,
) -> Result<CurveCopilotTensors> {
    let num_steps = query_times.len() as i64;
    let curve_window_len = curve_window.len() as i64;

    let context = Tensor::from_array((
        vec![
            1i64,
            CONTEXT_KEYFRAME_COUNT as i64,
            CONTEXT_FEATURE_DIM as i64,
        ],
        context.to_vec(),
    ))?;
    let property_type = Tensor::from_array((vec![1i64], vec![property_type_id as i64]))?;
    let topology = Tensor::from_array((
        vec![1i64, TOPOLOGY_FEATURE_DIM as i64],
        topology_features.to_vec(),
    ))?;
    let bone_name = Tensor::from_array((
        vec![1i64, BONE_NAME_TOKEN_DIM as i64],
        bone_name_tokens.to_vec(),
    ))?;
    let query_times_tensor = Tensor::from_array((vec![1i64, num_steps], query_times.to_vec()))?;
    let curve_window_tensor =
        Tensor::from_array((vec![1i64, curve_window_len], curve_window.to_vec()))?;

    let bone_context_keyframes = Tensor::from_array((
        vec![
            1i64,
            BONE_CONTEXT_N_MAX as i64,
            CONTEXT_KEYFRAME_COUNT as i64,
            CONTEXT_FEATURE_DIM as i64,
        ],
        bone_context.keyframes.to_vec(),
    ))?;
    let bone_context_topology = Tensor::from_array((
        vec![1i64, BONE_CONTEXT_N_MAX as i64, TOPOLOGY_FEATURE_DIM as i64],
        bone_context.topology.to_vec(),
    ))?;
    let bone_context_rest_positions = Tensor::from_array((
        vec![
            1i64,
            BONE_CONTEXT_N_MAX as i64,
            BONE_CONTEXT_REST_POSITION_DIM as i64,
        ],
        bone_context.rest_positions.to_vec(),
    ))?;
    let bone_context_mask = Tensor::from_array((
        vec![1i64, BONE_CONTEXT_N_MAX as i64],
        bone_context.mask.to_vec(),
    ))?;

    Ok(CurveCopilotTensors {
        context,
        property_type,
        topology,
        bone_name,
        query_times: query_times_tensor,
        curve_window: curve_window_tensor,
        bone_context_keyframes,
        bone_context_topology,
        bone_context_rest_positions,
        bone_context_mask,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_bone_context() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<bool>) {
        let n = BONE_CONTEXT_N_MAX;
        let kf_len = n * CONTEXT_KEYFRAME_COUNT * CONTEXT_FEATURE_DIM;
        let topo_len = n * TOPOLOGY_FEATURE_DIM;
        let rest_len = n * BONE_CONTEXT_REST_POSITION_DIM;
        (
            vec![0.0; kf_len],
            vec![0.0; topo_len],
            vec![0.0; rest_len],
            vec![false; n],
        )
    }

    #[test]
    fn build_curve_copilot_tensors_accepts_full_inputs() {
        let context = vec![0.0_f32; CONTEXT_KEYFRAME_COUNT * CONTEXT_FEATURE_DIM];
        let topology = vec![0.0_f32; TOPOLOGY_FEATURE_DIM];
        let tokens = vec![0_i64; BONE_NAME_TOKEN_DIM];
        let query_times = vec![0.0_f32; 1];
        let curve_window = vec![0.0_f32; 32];
        let (kf, topo, rest, mask) = zero_bone_context();

        let result = build_curve_copilot_tensors(
            &context,
            0,
            &topology,
            &tokens,
            &query_times,
            &curve_window,
            BoneContextInput {
                keyframes: &kf,
                topology: &topo,
                rest_positions: &rest,
                mask: &mask,
            },
        );

        assert!(result.is_ok());
    }
}
