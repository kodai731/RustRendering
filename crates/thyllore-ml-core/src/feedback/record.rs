use std::path::Path;

use anyhow::Context;
use serde::Serialize;
use sha2::{Digest, Sha256};
use thyllore_anim_core::editable::PropertyType;

/// Schema SSoT, shared with the Blender addon via the pybindings.
pub const FEEDBACK_SCHEMA_VERSION: &str = "curve_copilot_feedback/v1";

const TS_GRANULARITY_SECONDS: u64 = 86_400;
const QUANTIZE_FACTOR: f64 = 1.0e4;
const AMPLITUDE_EPSILON: f64 = 1.0e-9;

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct FeedbackChannel {
    pub kind: String,
    pub array_index: u32,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct FeedbackFps {
    pub scene: f64,
    pub deploy: f64,
    pub frame_step: f64,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct FeedbackRecord {
    pub schema: String,
    pub model_hash: String,
    pub channel: FeedbackChannel,
    pub fps: FeedbackFps,
    pub context: Vec<f64>,
    pub prediction: Vec<f64>,
    pub ground_truth: Option<Vec<f64>>,
    pub signal: String,
    pub record_id: String,
    pub revision: u32,
    pub ts: u64,
}

pub struct FeedbackRecordInputs<'a> {
    pub model_hash: &'a str,
    pub channel_kind: &'a str,
    pub array_index: u32,
    pub scene_fps: f64,
    pub deploy_fps: f64,
    pub frame_step: f64,
    pub origin_value: f64,
    pub context: &'a [f64],
    pub prediction: &'a [f64],
    pub ground_truth: Option<Vec<f64>>,
    pub signal: &'a str,
    pub record_id: &'a str,
    pub revision: u32,
    pub ts: u64,
}

/// Builds one anonymized feedback record. `context`/`prediction` are absolute
/// curve values; `ground_truth` is expected to be origin-relative already
/// (sampled that way by the caller).
///
/// The record is made non-reconstructable: values are stored relative to
/// `origin_value`, divided by the context's peak amplitude (the scale is NOT
/// transmitted, so real units and magnitudes are lost), quantized, and `ts`
/// is coarsened to day granularity so records cannot be re-correlated into
/// runs or bones. Only the normalized curve shape needed for training remains.
///
/// `record_id` + `revision` let one prediction be re-sent with an updated
/// ground truth after each save; training keeps the highest revision per id.
pub fn build_feedback_record(inputs: FeedbackRecordInputs<'_>) -> FeedbackRecord {
    let origin = inputs.origin_value;
    let relative_context: Vec<f64> = inputs.context.iter().map(|value| value - origin).collect();
    let scale = amplitude_scale(&relative_context);

    FeedbackRecord {
        schema: FEEDBACK_SCHEMA_VERSION.to_string(),
        model_hash: inputs.model_hash.to_string(),
        channel: FeedbackChannel {
            kind: inputs.channel_kind.to_string(),
            array_index: inputs.array_index,
        },
        fps: FeedbackFps {
            scene: inputs.scene_fps,
            deploy: inputs.deploy_fps,
            frame_step: inputs.frame_step,
        },
        context: relative_context
            .iter()
            .map(|value| quantize(value / scale))
            .collect(),
        prediction: inputs
            .prediction
            .iter()
            .map(|value| quantize((value - origin) / scale))
            .collect(),
        ground_truth: inputs
            .ground_truth
            .map(|values| values.iter().map(|value| quantize(value / scale)).collect()),
        signal: inputs.signal.to_string(),
        record_id: inputs.record_id.to_string(),
        revision: inputs.revision,
        ts: inputs.ts - inputs.ts % TS_GRANULARITY_SECONDS,
    }
}

fn amplitude_scale(relative_context: &[f64]) -> f64 {
    let peak = relative_context
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    if peak < AMPLITUDE_EPSILON {
        1.0
    } else {
        peak
    }
}

fn quantize(value: f64) -> f64 {
    (value * QUANTIZE_FACTOR).round() / QUANTIZE_FACTOR
}

/// Bone names never leave the engine; only the channel kind and axis index
/// are transmitted.
pub fn channel_for_property_type(property_type: PropertyType) -> FeedbackChannel {
    let (kind, array_index) = match property_type {
        PropertyType::TranslationX => ("location", 0),
        PropertyType::TranslationY => ("location", 1),
        PropertyType::TranslationZ => ("location", 2),
        PropertyType::RotationX => ("rotation_euler", 0),
        PropertyType::RotationY => ("rotation_euler", 1),
        PropertyType::RotationZ => ("rotation_euler", 2),
        PropertyType::ScaleX => ("scale", 0),
        PropertyType::ScaleY => ("scale", 1),
        PropertyType::ScaleZ => ("scale", 2),
    };
    FeedbackChannel {
        kind: kind.to_string(),
        array_index,
    }
}

pub fn hash_model_file(path: &Path) -> anyhow::Result<String> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read model file {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs<'a>(context: &'a [f64], prediction: &'a [f64]) -> FeedbackRecordInputs<'a> {
        FeedbackRecordInputs {
            model_hash: "abc123",
            channel_kind: "location",
            array_index: 1,
            scene_fps: 24.0,
            deploy_fps: 60.0,
            frame_step: 0.4,
            origin_value: 2.0,
            context,
            prediction,
            ground_truth: Some(vec![0.5, -0.25]),
            signal: "ignored",
            record_id: "rec-1",
            revision: 0,
            ts: 1_752_200_000,
        }
    }

    #[test]
    fn record_stores_origin_relative_values() {
        let record = build_feedback_record(inputs(&[2.0, 3.0], &[4.0, 1.0]));

        assert_eq!(record.schema, FEEDBACK_SCHEMA_VERSION);
        assert_eq!(record.context, vec![0.0, 1.0]);
        assert_eq!(record.prediction, vec![2.0, -1.0]);
        assert_eq!(record.ground_truth, Some(vec![0.5, -0.25]));
    }

    #[test]
    fn record_normalizes_amplitude_without_transmitting_scale() {
        let record = build_feedback_record(inputs(&[2.0, 6.0], &[4.0, 0.0]));

        assert_eq!(record.context, vec![0.0, 1.0]);
        assert_eq!(record.prediction, vec![0.5, -0.5]);
        assert_eq!(record.ground_truth, Some(vec![0.125, -0.0625]));
    }

    #[test]
    fn flat_context_falls_back_to_unit_scale() {
        let record = build_feedback_record(inputs(&[2.0, 2.0], &[3.0, 1.0]));

        assert_eq!(record.context, vec![0.0, 0.0]);
        assert_eq!(record.prediction, vec![1.0, -1.0]);
    }

    #[test]
    fn timestamp_is_coarsened_to_day_granularity() {
        let record = build_feedback_record(inputs(&[2.0], &[2.0]));
        assert_eq!(record.ts, 1_752_192_000);
        assert_eq!(record.ts % 86_400, 0);
    }

    #[test]
    fn record_serializes_with_schema_field_order() {
        let record = build_feedback_record(inputs(&[2.0], &[2.0]));
        let json = serde_json::to_string(&record).expect("serialize");

        let expected = concat!(
            "{\"schema\":\"curve_copilot_feedback/v1\",\"model_hash\":\"abc123\",",
            "\"channel\":{\"kind\":\"location\",\"array_index\":1},",
            "\"fps\":{\"scene\":24.0,\"deploy\":60.0,\"frame_step\":0.4},",
            "\"context\":[0.0],\"prediction\":[0.0],",
            "\"ground_truth\":[0.5,-0.25],\"signal\":\"ignored\",",
            "\"record_id\":\"rec-1\",\"revision\":0,\"ts\":1752192000}"
        );
        assert_eq!(json, expected);
    }

    #[test]
    fn missing_ground_truth_serializes_as_null() {
        let mut record = build_feedback_record(inputs(&[], &[]));
        record.ground_truth = None;
        let json = serde_json::to_string(&record).expect("serialize");
        assert!(json.contains("\"ground_truth\":null"));
    }

    #[test]
    fn channel_mapping_covers_all_property_types() {
        let cases = [
            (PropertyType::TranslationX, "location", 0),
            (PropertyType::TranslationY, "location", 1),
            (PropertyType::TranslationZ, "location", 2),
            (PropertyType::RotationX, "rotation_euler", 0),
            (PropertyType::RotationY, "rotation_euler", 1),
            (PropertyType::RotationZ, "rotation_euler", 2),
            (PropertyType::ScaleX, "scale", 0),
            (PropertyType::ScaleY, "scale", 1),
            (PropertyType::ScaleZ, "scale", 2),
        ];
        for (property_type, kind, array_index) in cases {
            let channel = channel_for_property_type(property_type);
            assert_eq!(channel.kind, kind);
            assert_eq!(channel.array_index, array_index);
        }
    }

    #[test]
    fn hash_model_file_is_sha256_hex() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("model.onnx");
        std::fs::write(&path, b"dummy").expect("write");

        let hash = hash_model_file(&path).expect("hash");
        assert_eq!(
            hash,
            "b5a2c96250612366ea272ffac6d9744aaf4b45aacd96aa7cfcb931ee3b558259"
        );
    }
}
