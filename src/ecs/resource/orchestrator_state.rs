//! Orchestrator ECS resource.

use std::path::Path;

use thyllore_ml_core::sentence_encoder::SentenceEncoder;

use crate::orchestrator::components::route::{OrchestratorMode, Route};
use crate::orchestrator::systems::polarity_tiebreak::embedded_exemplars_sha256;
use crate::orchestrator::systems::router::{ExemplarIndex, RouterThresholds};

/// Default model directory for the orchestrator runtime.
pub const ROUTER_MODEL_DIR: &str = "models/gemma/setfit-3ep-p2";

/// Raw encoder model directory (for escape detection).
pub const RAW_ENCODER_DIR: &str = "models/gemma/e5-raw";

#[derive(Clone, Debug)]
pub enum CommandFeedback {
    Router(crate::orchestrator::systems::resolution::OrchestratorFeedback),
    Report(String),
    Executed(String),
    DispatchError(String),
    Unavailable(String),
}

/// The loaded orchestrator runtime (encoder + exemplar index).
pub struct OrchestratorRuntime {
    pub encoder: SentenceEncoder,
    pub index: ExemplarIndex,
    pub raw_encoder: SentenceEncoder,
    pub raw_index: ExemplarIndex,
}

/// The runtime's current state.
pub enum RuntimeSlot {
    Uninitialized,
    Ready(Box<OrchestratorRuntime>),
    Failed(String),
}

/// Orchestrator state stored as an ECS resource.
pub struct OrchestratorState {
    pub mode: OrchestratorMode,
    pub submitted_utterance: Option<String>,
    pub confirm_response: Option<bool>,
    pub clarify_choice: Option<Route>,
    pub last_utterance: Option<String>,
    pub pending: Option<(crate::orchestrator::components::tool_call::ToolCall, crate::orchestrator::systems::resolution::ConfirmReason)>,
    pub feedback: Option<CommandFeedback>,
    pub confirm_all: bool,
    pub thresholds: RouterThresholds,
    pub runtime: RuntimeSlot,
    pub motion_seed_counters: std::collections::HashMap<crate::orchestrator::components::tool_call::MotionCategory, usize>,
}

impl Default for OrchestratorState {
    fn default() -> Self {
        Self {
            mode: OrchestratorMode::ReadOnly,
            submitted_utterance: None,
            confirm_response: None,
            clarify_choice: None,
            last_utterance: None,
            pending: None,
            feedback: None,
            confirm_all: true,
            thresholds: RouterThresholds {
                tau_reject: crate::orchestrator::systems::router::TUNED_TAU_REJECT,
                delta: crate::orchestrator::systems::router::TUNED_DELTA,
                tau_confirm: crate::orchestrator::systems::router::TUNED_TAU_CONFIRM,
                tau_raw: crate::orchestrator::systems::router::TUNED_TAU_RAW,
            },
            runtime: RuntimeSlot::Uninitialized,
            motion_seed_counters: std::collections::HashMap::new(),
        }
    }
}

/// Validate that the router index and polarity table were generated from the same exemplars.jsonl.
///
/// Returns Ok(()) if both hashes match or if either is None (old artifact compatibility).
/// Returns Err with a message if both are Some but differ.
pub fn validate_artifact_consistency(index_hash: Option<&str>, table_hash: Option<&str>) -> Result<(), String> {
    match (index_hash, table_hash) {
        (Some(ih), Some(th)) if ih != th => Err(
            "router index and polarity table were generated from different exemplars.jsonl \
            — re-run AnimationModelTraining scripts/orchestrator_router/export_router_index.py"
                .to_string(),
        ),
        _ => Ok(()),
    }
}
/// Load the orchestrator runtime from a model directory.
pub fn load_runtime(model_dir: &Path) -> Result<OrchestratorRuntime, String> {
    let encoder = SentenceEncoder::from_model_dir(model_dir)
        .map_err(|e| format!("failed to load sentence encoder: {}", e))?;

    let manifest_path = model_dir.join("router_index.json");
    let manifest_json = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("failed to read {}: {}", manifest_path.display(), e))?;

    let vector_path = model_dir.join("router_index.f32");
    let vector_bytes = std::fs::read(&vector_path)
        .map_err(|e| format!("failed to read {}: {}", vector_path.display(), e))?;

    let index = ExemplarIndex::from_export(&manifest_json, &vector_bytes)
        .map_err(|e| format!("failed to load exemplar index: {}", e))?;

    // Staleness check: if both index and embedded polarity table carry exemplars_sha256,
    // they must match — otherwise one artifact is stale.
    validate_artifact_consistency(index.exemplars_sha256(), embedded_exemplars_sha256().as_deref())?;

    // Load raw encoder for escape detection
    let raw_encoder = SentenceEncoder::from_model_dir(Path::new(RAW_ENCODER_DIR))
        .map_err(|e| format!("failed to load raw sentence encoder: {}", e))?;

    // Load raw index from model_dir/raw_index.json and model_dir/raw_index.f32
    let raw_manifest_path = model_dir.join("raw_index.json");
    let raw_manifest_json = std::fs::read_to_string(&raw_manifest_path)
        .map_err(|e| format!("failed to read {}: {}", raw_manifest_path.display(), e))?;

    let raw_vector_path = model_dir.join("raw_index.f32");
    let raw_vector_bytes = std::fs::read(&raw_vector_path)
        .map_err(|e| format!("failed to read {}: {}", raw_vector_path.display(), e))?;

    let raw_index = ExemplarIndex::from_export(&raw_manifest_json, &raw_vector_bytes)
        .map_err(|e| format!("failed to load raw exemplar index: {}", e))?;

    // Validate that raw_index exemplars_sha256 matches setfit index
    validate_artifact_consistency(raw_index.exemplars_sha256(), index.exemplars_sha256())?;

    Ok(OrchestratorRuntime { encoder, index, raw_encoder, raw_index })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_artifact_consistency_both_some_match() {
        let result = validate_artifact_consistency(Some("abc123"), Some("abc123"));
        assert!(result.is_ok(), "matching hashes should be Ok");
    }

    #[test]
    fn test_validate_artifact_consistency_both_some_mismatch() {
        let result = validate_artifact_consistency(Some("abc123"), Some("def456"));
        assert!(result.is_err(), "mismatching hashes should be Err");
        let err = result.unwrap_err();
        assert!(
            err.contains("different exemplars.jsonl"),
            "error message should mention different exemplars.jsonl"
        );
    }

    #[test]
    fn test_validate_artifact_consistency_one_none() {
        let result1 = validate_artifact_consistency(Some("abc123"), None);
        assert!(result1.is_ok(), "one None should be Ok (old artifact compatibility)");

        let result2 = validate_artifact_consistency(None, Some("abc123"));
        assert!(result2.is_ok(), "one None should be Ok (old artifact compatibility)");
    }

    #[test]
    fn test_validate_artifact_consistency_both_none() {
        let result = validate_artifact_consistency(None, None);
        assert!(result.is_ok(), "both None should be Ok (old artifact compatibility)");
    }
}
