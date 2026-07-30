//! Helm ECS resource.

use std::env;
use std::path::{Path, PathBuf};

use thyllore_ml_core::model_path::{EXPORTS_SUBDIR, SHARED_DATA_ENV_VAR};
use thyllore_ml_core::sentence_encoder::SentenceEncoder;

use crate::helm::components::route::{HelmMode, Route};
use crate::helm::systems::polarity_tiebreak::embedded_exemplars_sha256;
use crate::helm::systems::router::{ExemplarIndex, RouterThresholds};

/// Default model directory for the helm runtime.
pub const ROUTER_MODEL_DIR: &str = "models/gemma/setfit-3ep-p2";

/// Raw encoder model directory (for escape detection).
pub const RAW_ENCODER_DIR: &str = "models/gemma/e5-raw";

/// Exports bundle directory name inside SharedData/exports.
pub const EXPORTS_BUNDLE_DIR: &str = "helm_router_20260728";

/// Select the router model directory based on shared data availability.
///
/// If `shared_data_dir` is Some and `shared_data_dir/exports/helm_router_20260728/setfit-3ep-p2`
/// is a directory, return that path. Otherwise, return `PathBuf::from(ROUTER_MODEL_DIR)`.
pub fn select_router_model_dir(shared_data_dir: Option<&std::path::Path>) -> std::path::PathBuf {
    if let Some(shared) = shared_data_dir {
        let bundle_path = shared
            .join(EXPORTS_SUBDIR)
            .join(EXPORTS_BUNDLE_DIR)
            .join("setfit-3ep-p2");
        if bundle_path.is_dir() {
            return bundle_path;
        }
    }
    PathBuf::from(ROUTER_MODEL_DIR)
}

/// Resolve the router model directory by reading `THYLLORE_SHARED_DATA_DIR` from the environment
/// and calling `select_router_model_dir`.
pub fn resolve_router_model_dir() -> std::path::PathBuf {
    let shared_data_dir = env::var(SHARED_DATA_ENV_VAR)
        .ok()
        .map(std::path::PathBuf::from);
    select_router_model_dir(shared_data_dir.as_deref())
}

#[derive(Clone, Debug)]
pub enum CommandFeedback {
    Router(crate::helm::systems::resolution::HelmFeedback),
    Report(String),
    Executed(String),
    DispatchError(String),
    Unavailable(String),
}

/// The loaded helm runtime (encoder + exemplar index).
pub struct HelmRuntime {
    pub encoder: SentenceEncoder,
    pub index: ExemplarIndex,
    pub raw_encoder: SentenceEncoder,
    pub raw_index: ExemplarIndex,
}

/// The runtime's current state.
pub enum RuntimeSlot {
    Uninitialized,
    Ready(Box<HelmRuntime>),
    Failed(String),
}

/// Helm state stored as an ECS resource.
pub struct HelmState {
    pub mode: HelmMode,
    pub submitted_utterance: Option<String>,
    pub confirm_response: Option<bool>,
    pub clarify_choice: Option<Route>,
    pub last_utterance: Option<String>,
    pub pending: Option<(
        crate::helm::components::tool_call::ToolCall,
        crate::helm::systems::resolution::ConfirmReason,
    )>,
    pub feedback: Option<CommandFeedback>,
    pub confirm_all: bool,
    pub thresholds: RouterThresholds,
    pub runtime: RuntimeSlot,
    pub motion_seed_counters:
        std::collections::HashMap<crate::helm::components::tool_call::MotionCategory, usize>,
    pub last_routed_tool: Option<String>,
    pub last_route_latency_ms: Option<f32>,
    pub last_runtime_load_ms: Option<f32>,
}

impl Default for HelmState {
    fn default() -> Self {
        Self {
            mode: HelmMode::ReadOnly,
            submitted_utterance: None,
            confirm_response: None,
            clarify_choice: None,
            last_utterance: None,
            pending: None,
            feedback: None,
            confirm_all: true,
            thresholds: RouterThresholds {
                tau_reject: crate::helm::systems::router::TUNED_TAU_REJECT,
                delta: crate::helm::systems::router::TUNED_DELTA,
                tau_confirm: crate::helm::systems::router::TUNED_TAU_CONFIRM,
                tau_raw: crate::helm::systems::router::TUNED_TAU_RAW,
                tau_raw_nearmiss: 0.85,
            },
            runtime: RuntimeSlot::Uninitialized,
            motion_seed_counters: std::collections::HashMap::new(),
            last_routed_tool: None,
            last_route_latency_ms: None,
            last_runtime_load_ms: None,
        }
    }
}

/// Validate that the router index and polarity table were generated from the same exemplars.jsonl.
///
/// Returns Ok(()) if both hashes match or if either is None (old artifact compatibility).
/// Returns Err with a message if both are Some but differ.
pub fn validate_artifact_consistency(
    index_hash: Option<&str>,
    table_hash: Option<&str>,
) -> Result<(), String> {
    match (index_hash, table_hash) {
        (Some(ih), Some(th)) if ih != th => Err(
            "router index and polarity table were generated from different exemplars.jsonl \
            — re-run AnimationModelTraining scripts/helm_router/export_router_index.py"
                .to_string(),
        ),
        _ => Ok(()),
    }
}

/// Load the helm runtime from a model directory.
pub fn load_runtime(model_dir: &Path) -> Result<HelmRuntime, String> {
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
    validate_artifact_consistency(
        index.exemplars_sha256(),
        embedded_exemplars_sha256().as_deref(),
    )?;

    // Load raw encoder for escape detection: prefer bundle sibling (model_dir/../e5-raw) if it
    // exists, otherwise fall back to the fixed RAW_ENCODER_DIR.
    let raw_encoder_dir = model_dir
        .parent()
        .map(|p| p.join("e5-raw"))
        .filter(|p| p.is_dir());
    let raw_encoder_path: &Path = match raw_encoder_dir {
        Some(ref p) => p.as_path(),
        None => Path::new(RAW_ENCODER_DIR),
    };
    let raw_encoder = SentenceEncoder::from_model_dir(raw_encoder_path)
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

    Ok(HelmRuntime {
        encoder,
        index,
        raw_encoder,
        raw_index,
    })
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
        assert!(
            result1.is_ok(),
            "one None should be Ok (old artifact compatibility)"
        );

        let result2 = validate_artifact_consistency(None, Some("abc123"));
        assert!(
            result2.is_ok(),
            "one None should be Ok (old artifact compatibility)"
        );
    }

    #[test]
    fn test_validate_artifact_consistency_both_none() {
        let result = validate_artifact_consistency(None, None);
        assert!(
            result.is_ok(),
            "both None should be Ok (old artifact compatibility)"
        );
    }

    #[test]
    fn test_select_router_model_dir_none_returns_default() {
        let path = select_router_model_dir(None);
        assert_eq!(path, PathBuf::from(ROUTER_MODEL_DIR));
    }

    #[test]
    fn test_select_router_model_dir_non_existent_returns_default() {
        let non_existent = std::path::Path::new("/non/existent/path");
        let path = select_router_model_dir(Some(non_existent));
        assert_eq!(path, PathBuf::from(ROUTER_MODEL_DIR));
    }

    #[test]
    fn test_select_router_model_dir_bundle_exists_returns_bundle() {
        let temp_dir = std::env::temp_dir().join("thyllore_test_select_router_model_dir");
        let bundle_path = temp_dir
            .join(EXPORTS_SUBDIR)
            .join(EXPORTS_BUNDLE_DIR)
            .join("setfit-3ep-p2");
        std::fs::create_dir_all(&bundle_path).expect("failed to create temp bundle dir");

        let path = select_router_model_dir(Some(&temp_dir));
        assert_eq!(path, bundle_path);

        std::fs::remove_dir_all(&temp_dir).expect("failed to remove temp dir");
    }
}
