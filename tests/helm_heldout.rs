//! Held-out evaluation of the helm router on 128 samples.
//!
//! Loads JSONL from `THYLLORE_HELM_HELDOUT` (each line `{lang, utterance, tool, args}`),
//! loads the runtime from `THYLLORE_ROUTER_MODEL_DIR` via `load_runtime`, and measures
//! route accuracy against expected routes derived from the index.
//!
//! Scoring matches `src/helm/systems/router.rs` exactly: pre-threshold ranking
//! (`rank_routes` + polarity tie-break) gives predicted_route and top_score;
//! raw encoder + raw index gives raw_top_score (top-1 cosine similarity).
//!
//! Metrics match `AnimationModelTraining scripts/helm_router/eval_router.py` summarize()
//! with tau1=0.93, tau2=0.90: routed = expected != escape, correct = predicted == expected,
//! escape_rejected = escape cases where NOT(top_score >= 0.93 && raw_top_score >= 0.90),
//! retained = correct among routed, post_gate_accuracy = correct / retained.
//!
//! Requires `THYLLORE_ROUTER_MODEL_DIR` and `THYLLORE_HELM_HELDOUT`. Skips if either env
//! is missing (developer-machine test, not CI).

use std::path::PathBuf;

use serde::Deserialize;
use thyllore_animation::ecs::resource::{load_runtime, HelmRuntime};
use thyllore_animation::helm::components::route::HelmMode;
use thyllore_animation::helm::systems::normalize::normalize_utterance;
use thyllore_animation::helm::systems::router::{
    rank_routes, route_utterance, RouterDecision, RouterThresholds, RoutingRequest,
};

const MODEL_DIR_ENV_VAR: &str = "THYLLORE_ROUTER_MODEL_DIR";
const HELDOUT_ENV_VAR: &str = "THYLLORE_HELM_HELDOUT";

#[derive(Deserialize)]
struct HeldoutRow {
    #[allow(dead_code)]
    lang: String,
    utterance: String,
    tool: String,
    #[allow(dead_code)]
    args: serde_json::Value,
}

fn resolve_model_dir() -> Option<PathBuf> {
    let model_dir = PathBuf::from(std::env::var(MODEL_DIR_ENV_VAR).ok()?);
    model_dir.exists().then_some(model_dir)
}

fn resolve_heldout_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(HELDOUT_ENV_VAR).ok()?);
    path.exists().then_some(path)
}

/// Derive the expected route id from a held-out row by matching against index routes.
///
/// Find candidates in index where id == tool or id starts with "tool:".
/// - 0 candidates: expected = "escape"
/// - 1 candidate: expected = that id
/// - >1 candidates: find unique id where any string value in args matches the suffix after "tool:"; panic if not unique
fn derive_expected(index_route_ids: &[String], tool: &str, args: &serde_json::Value) -> String {
    let candidates: Vec<&String> = index_route_ids
        .iter()
        .filter(|id| id.as_str() == tool || id.starts_with(&format!("{}:", tool)))
        .collect();

  if candidates.is_empty() {
        return "__escape__".to_string();
    }

    if candidates.len() == 1 {
        return candidates[0].clone();
    }

    // Multiple candidates: find unique one where any string value in args matches the suffix
   let matches: Vec<&String> = candidates
        .into_iter()
        .filter(|id| {
            if let Some(suffix) = id.strip_prefix(&format!("{}:", tool)) {
                args_contains_value(args, suffix)
            } else {
                // Exact match (no colon suffix) — only valid if args is empty
                *id == tool && is_empty_args(args)
            }
        })
        .collect();

    if matches.len() != 1 {
        panic!(
            "ambiguous expected route for tool={:?} args={:?}: candidates={:?}, matches={:?}",
            tool, args, index_route_ids, matches
        );
    }

    matches[0].clone()
}

/// Check if any string value in the args JSON equals the expected suffix.
fn args_contains_value(args: &serde_json::Value, expected: &str) -> bool {
    match args {
        serde_json::Value::String(s) => s == expected,
        serde_json::Value::Object(map) => map.values().any(|v| {
            if let serde_json::Value::String(s) = v {
                s == expected
            } else {
                false
            }
        }),
        _ => false,
    }
}

/// Check if args is effectively empty (empty object or null).
fn is_empty_args(args: &serde_json::Value) -> bool {
    match args {
        serde_json::Value::Object(map) => map.is_empty(),
        serde_json::Value::Null => true,
        _ => false,
    }
}

#[test]
fn helm_heldout_evaluation() {
    let Some(model_dir) = resolve_model_dir() else {
        eprintln!(
            "Skipping: set {MODEL_DIR_ENV_VAR} to a model directory prepared by \
             scripts/helm_router/export_router_index.py"
        );
        return;
    };

    let Some(heldout_path) = resolve_heldout_path() else {
        eprintln!(
            "Skipping: set {HELDOUT_ENV_VAR} to the held-out jsonl (e.g. \
             scripts/helm_router/heldout.jsonl)"
        );
        return;
    };
    // Load runtime (encoder + index + raw encoder + raw index)
    let mut runtime: HelmRuntime =
        load_runtime(&model_dir).expect("load_runtime must succeed");

    // Load held-out data
    let jsonl = std::fs::read_to_string(&heldout_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", heldout_path.display(), e));
    let rows: Vec<HeldoutRow> = jsonl
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line).unwrap_or_else(|e| {
                panic!("failed to parse held-out line: {}\nline: {}", e, line)
            })
        })
        .collect();

    let total = rows.len();
    assert_eq!(total, 128, "expected 128 held-out samples, got {}", total);

    // Get route IDs from index for expected derivation
    let query_vector = vec![0.0f32; runtime.index.dimensions()];
    let ranked = rank_routes(&runtime.index, &query_vector, HelmMode::AllowEdit);
    let index_route_ids: Vec<String> = ranked.into_iter().map(|(route, _)| route.id()).collect();
    let thresholds = RouterThresholds {
        tau_reject: 0.93,
        delta: 0.0,
        tau_confirm: 0.0,
        tau_raw: 0.90,
        tau_raw_nearmiss: 0.85,
    };

    let mut correct = 0usize;
    let mut routed_count = 0usize;
    let mut escape_total = 0usize;
    let mut escape_rejected = 0usize;
    let mut retained_correct = 0usize;
    let mut retained_total = 0usize;
    let mut nearmiss_rescued = 0usize;

    for (i, row) in rows.iter().enumerate() {
        let expected = derive_expected(&index_route_ids, &row.tool, &row.args);
        let is_escape = expected == "__escape__";

        if is_escape {
            escape_total += 1;
        } else {
            routed_count += 1;
        }

        // Normalize and encode
        let normalized = normalize_utterance(&row.utterance);
        let query = runtime
            .encoder
            .encode(&row.utterance)
            .expect("encoder must produce a vector");

        // Get raw top score from raw encoder + raw index
        let raw_query = runtime
            .raw_encoder
            .encode(&row.utterance)
            .expect("raw encoder must produce a vector");
        let raw_top_score = compute_raw_top_score(&runtime.raw_index, &raw_query);

        // Route with pre-threshold ranking (same as route_utterance)
        let decision = route_utterance(
            RoutingRequest {
                utterance: &normalized,
                query_vector: &query,
                mode: HelmMode::AllowEdit,
                raw_top_score: Some(raw_top_score),
            },
            &runtime.index,
            thresholds,
        );

        let (predicted_route, top_score) = match decision {
            RouterDecision::Accept { route, score, .. } => (route.id(), score),
            RouterDecision::Reject { best, score } => (best.id(), score),
            RouterDecision::Clarify { candidates, .. } => {
                let (route, score) = candidates[0];
                (route.id(), score)
            }
            RouterDecision::NoCandidate => ("__escape__".to_string(), 0.0),
        };

        let is_correct = predicted_route == expected;
        if is_correct {
            correct += 1;
            if top_score >= 0.93 && raw_top_score >= 0.85 && raw_top_score < 0.90 {
                nearmiss_rescued += 1;
            }
        }
        // Escape rejection: expected == escape AND NOT(top_score >= 0.93 && raw_top_score >= 0.90)
        if is_escape {
            let gate_passed = top_score >= 0.93 && raw_top_score >= 0.90;
            if !gate_passed {
                escape_rejected += 1;
            }
        }

        // Retained: correct among routed (non-escape) cases that pass the gate
        if !is_escape {
            let gate_passed = top_score >= 0.93 && raw_top_score >= 0.90;
            if gate_passed {
                retained_total += 1;
                if is_correct {
                    retained_correct += 1;
                }
            }
        }

        #[allow(unused_variables)]
        let debug = std::env::var("THYLLORE_HELM_DEBUG").is_ok();
        if debug {
            eprintln!(
                "[{:3}] expected={} predicted={} top={:.4} raw={:.4} correct={}",
                i + 1,
                expected,
                predicted_route,
                top_score,
                raw_top_score,
                is_correct
            );
        }
    }

    // Compute metrics
    let post_gate_accuracy = if retained_total > 0 {
        retained_correct as f32 / retained_total as f32
    } else {
        0.0
    };

    // Print summary
    println!("Held-out evaluation ({} samples):", total);
    println!("  routed:          {}/{}", routed_count, total);
    println!("  correct:         {}/{}", correct, routed_count);
    println!("  escape total:    {}", escape_total);
    println!("  escape rejected: {}/{}", escape_rejected, escape_total);
    println!("  retained:        {}/{}", retained_correct, retained_total);
    println!("  post-gate acc:   {:.4}", post_gate_accuracy);
    println!("  nearmiss rescued: {}/{}", nearmiss_rescued, routed_count);

    // Assertions (based on setfit-3ep-p2 p2_final_heldout.json)
    assert_eq!(
        routed_count, 116,
        "expected 116 routed samples, got {}",
        routed_count
    );
    assert_eq!(
        correct, 100,
        "expected 100 correct routes, got {}",
        correct
    );
    assert_eq!(
        escape_rejected, 10,
        "expected 10 escape rejected, got {}",
        escape_rejected
    );
    assert_eq!(
        retained_correct, 79,
        "expected 79 retained correct, got {}",
        retained_correct
    );
}

/// Compute the top-1 cosine similarity between query and all vectors in the index.
fn compute_raw_top_score(
    index: &thyllore_animation::helm::systems::router::ExemplarIndex,
    query: &[f32],
) -> f32 {
    let ranked = rank_routes(index, query, HelmMode::AllowEdit);
    ranked.first().map(|(_, score)| *score).unwrap_or(f32::NEG_INFINITY)
}
