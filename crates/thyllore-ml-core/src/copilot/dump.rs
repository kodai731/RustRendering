use std::fs::{create_dir_all, File};
use std::path::{Path, PathBuf};

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4};
use ndarray_npy::NpzWriter;

use super::input::{
    BONE_CONTEXT_N_MAX, BONE_CONTEXT_REST_POSITION_DIM, BONE_NAME_TOKEN_DIM, CONTEXT_FEATURE_DIM,
    CONTEXT_KEYFRAME_COUNT, TOPOLOGY_FEATURE_DIM,
};

pub struct CurveCopilotInferenceDump<'a> {
    pub context_keyframes: &'a [f32],
    pub property_type_id: u32,
    pub topology_features: &'a [f32],
    pub bone_name_tokens: &'a [i64],
    pub query_times: &'a [f32],
    pub curve_window: &'a [f32],
    pub bone_context_keyframes: &'a [f32],
    pub bone_context_topology: &'a [f32],
    pub bone_context_rest_positions: &'a [f32],
    pub bone_context_mask: &'a [bool],
    pub onnx_predictions: &'a [f32],
    pub onnx_confidences: &'a [f32],
    pub onnx_prediction_steps: i64,
    pub onnx_prediction_features_per_step: i64,
    pub curve_mean: f32,
    pub curve_std: f32,
    pub clip_duration: f32,
    pub denorm_query_times: &'a [f32],
    pub real_curve_samples: &'a [f32],
    pub nearest_keyframe_times: &'a [f32],
    pub nearest_keyframe_values: &'a [f32],
}

pub struct RawFutureInferenceDump<'a> {
    pub context: &'a [f32],
    pub future: &'a [f32],
    pub reveal_mask: &'a [bool],
    pub mean_curve: &'a [f32],
    pub fps: f32,
    pub anchor_time: f32,
}

pub fn dump_rawfuture_inference(
    dump: &RawFutureInferenceDump<'_>,
    output_dir: &Path,
) -> Result<PathBuf> {
    create_dir_all(output_dir)?;
    let timestamp = chrono::Local::now().format("%Y%m%d-%H%M%S-%3f").to_string();
    let path = output_dir.join(format!("{}-rawfuture-inference.npz", timestamp));

    let mut npz = NpzWriter::new(File::create(&path)?);

    npz.add_array("context", &Array1::from_vec(dump.context.to_vec()))?;
    npz.add_array("future", &Array1::from_vec(dump.future.to_vec()))?;
    let reveal_u8: Vec<u8> = dump.reveal_mask.iter().map(|b| *b as u8).collect();
    npz.add_array("reveal_mask", &Array1::from_vec(reveal_u8))?;
    npz.add_array("mean_curve", &Array1::from_vec(dump.mean_curve.to_vec()))?;
    npz.add_array("fps", &Array1::from_vec(vec![dump.fps]))?;
    npz.add_array("anchor_time", &Array1::from_vec(vec![dump.anchor_time]))?;

    npz.finish()?;
    Ok(path)
}

pub fn dump_curve_copilot_inference(
    dump: &CurveCopilotInferenceDump<'_>,
    output_dir: &Path,
) -> Result<PathBuf> {
    create_dir_all(output_dir)?;
    let timestamp = chrono::Local::now().format("%Y%m%d-%H%M%S-%3f").to_string();
    let path = output_dir.join(format!("{}-inference.npz", timestamp));

    let mut npz = NpzWriter::new(File::create(&path)?);

    let context = Array3::from_shape_vec(
        (1, CONTEXT_KEYFRAME_COUNT, CONTEXT_FEATURE_DIM),
        dump.context_keyframes.to_vec(),
    )?;
    npz.add_array("context_keyframes", &context)?;

    let property_type = Array1::from_vec(vec![dump.property_type_id as i64]);
    npz.add_array("property_type", &property_type)?;

    let topology =
        Array2::from_shape_vec((1, TOPOLOGY_FEATURE_DIM), dump.topology_features.to_vec())?;
    npz.add_array("topology_features", &topology)?;

    let bone_name =
        Array2::from_shape_vec((1, BONE_NAME_TOKEN_DIM), dump.bone_name_tokens.to_vec())?;
    npz.add_array("bone_name_tokens", &bone_name)?;

    let num_steps = dump.query_times.len();
    let query_times = Array2::from_shape_vec((1, num_steps), dump.query_times.to_vec())?;
    npz.add_array("query_times", &query_times)?;

    let curve_window =
        Array2::from_shape_vec((1, dump.curve_window.len()), dump.curve_window.to_vec())?;
    npz.add_array("curve_window", &curve_window)?;

    let bone_context_keyframes = Array4::from_shape_vec(
        (
            1,
            BONE_CONTEXT_N_MAX,
            CONTEXT_KEYFRAME_COUNT,
            CONTEXT_FEATURE_DIM,
        ),
        dump.bone_context_keyframes.to_vec(),
    )?;
    npz.add_array("bone_context_keyframes", &bone_context_keyframes)?;

    let bone_context_topology = Array3::from_shape_vec(
        (1, BONE_CONTEXT_N_MAX, TOPOLOGY_FEATURE_DIM),
        dump.bone_context_topology.to_vec(),
    )?;
    npz.add_array("bone_topology_features", &bone_context_topology)?;

    let bone_rest_positions = Array3::from_shape_vec(
        (1, BONE_CONTEXT_N_MAX, BONE_CONTEXT_REST_POSITION_DIM),
        dump.bone_context_rest_positions.to_vec(),
    )?;
    npz.add_array("bone_rest_positions", &bone_rest_positions)?;

    let mask_as_u8: Vec<u8> = dump.bone_context_mask.iter().map(|b| *b as u8).collect();
    let bone_context_mask = Array2::from_shape_vec((1, BONE_CONTEXT_N_MAX), mask_as_u8)?;
    npz.add_array("bone_context_mask", &bone_context_mask)?;

    let onnx_predictions = Array3::from_shape_vec(
        (
            1,
            dump.onnx_prediction_steps as usize,
            dump.onnx_prediction_features_per_step as usize,
        ),
        dump.onnx_predictions.to_vec(),
    )?;
    npz.add_array("onnx_predictions", &onnx_predictions)?;

    let onnx_confidences = Array3::from_shape_vec(
        (1, dump.onnx_prediction_steps as usize, 1),
        dump.onnx_confidences.to_vec(),
    )?;
    npz.add_array("onnx_confidences", &onnx_confidences)?;

    let curve_mean = Array1::from_vec(vec![dump.curve_mean]);
    npz.add_array("curve_mean", &curve_mean)?;
    let curve_std = Array1::from_vec(vec![dump.curve_std]);
    npz.add_array("curve_std", &curve_std)?;
    let clip_duration = Array1::from_vec(vec![dump.clip_duration]);
    npz.add_array("clip_duration", &clip_duration)?;

    let denorm_query_times = Array1::from_vec(dump.denorm_query_times.to_vec());
    npz.add_array("denorm_query_times", &denorm_query_times)?;

    let real_curve_samples = Array1::from_vec(dump.real_curve_samples.to_vec());
    npz.add_array("real_curve_samples", &real_curve_samples)?;

    let nearest_keyframe_times = Array1::from_vec(dump.nearest_keyframe_times.to_vec());
    npz.add_array("nearest_keyframe_times", &nearest_keyframe_times)?;

    let nearest_keyframe_values = Array1::from_vec(dump.nearest_keyframe_values.to_vec());
    npz.add_array("nearest_keyframe_values", &nearest_keyframe_values)?;

    npz.finish()?;

    Ok(path)
}
