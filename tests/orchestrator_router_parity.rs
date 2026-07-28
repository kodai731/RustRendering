//! Holds the Rust router to the decisions the Python evaluation driver reached.
//!
//! The published route accuracy was measured by `AnimationModelTraining scripts/orchestrator_router/`, so
//! the Rust port is only as good as its agreement with it. Every step can differ
//! quietly — a tokenizer that adds different special tokens, pooling that counts
//! padding, an aggregation that averages instead of taking the best exemplar, a
//! tie-break that reads a different table — and each one would move the engine off
//! the number without breaking anything visible.
//!
//! `export_router_index.py` writes `router_parity.json` next to the model: the
//! route and score that driver produced for all 202 labelled utterances. This
//! replays them through the Rust encoder, the Rust ranker and the Rust tie-break.
//!
//! Requires THYLLORE_ROUTER_MODEL_DIR pointing at a model directory holding
//! `tokenizer.json`, `onnx/model.onnx` and the three files that export writes, plus
//! ORT_DYLIB_PATH resolving to the vendored runtime.
//!
//! A developer-machine test, not a CI one: it needs both the root crate (which CI
//! cannot compile, see `.claude/rules/testing.md`) and a trained encoder that is
//! not in the repository. Do not wire it into a workflow.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use thyllore_animation::orchestrator::components::route::{OrchestratorMode, Route};
use thyllore_animation::orchestrator::systems::router::{
    route_utterance, ExemplarIndex, RouterDecision, RouterThresholds, RoutingRequest,
};
use thyllore_ml_core::sentence_encoder::SentenceEncoder;

const MODEL_DIR_ENV_VAR: &str = "THYLLORE_ROUTER_MODEL_DIR";
const MANIFEST_FILENAME: &str = "router_index.json";
const VECTORS_FILENAME: &str = "router_index.f32";
const PARITY_FILENAME: &str = "router_parity.json";

/// Python runs onnxruntime from pip, this runs the vendored build, and the two
/// pick different kernels for the same graph. A tolerance this wide still rules
/// out every difference that is a porting mistake rather than f32 jitter.
const SCORE_TOLERANCE: f32 = 1e-3;

#[derive(Deserialize)]
struct ParityFixture {
    stage_a: String,
    exemplars: usize,
    cases: Vec<ParityCase>,
}

#[derive(Deserialize)]
struct ParityCase {
    dataset: String,
    utterance: String,
    route: String,
    score: f32,
}

struct Harness {
    encoder: SentenceEncoder,
    index: ExemplarIndex,
    fixture: ParityFixture,
}

fn resolve_model_dir() -> Option<PathBuf> {
    let model_dir = PathBuf::from(std::env::var(MODEL_DIR_ENV_VAR).ok()?);
    model_dir
        .join(PARITY_FILENAME)
        .exists()
        .then_some(model_dir)
}

fn load_index(model_dir: &Path) -> ExemplarIndex {
    let manifest = std::fs::read_to_string(model_dir.join(MANIFEST_FILENAME))
        .expect("router_index.json must be readable");
    let vectors =
        std::fs::read(model_dir.join(VECTORS_FILENAME)).expect("router_index.f32 must be readable");

    ExemplarIndex::from_export(&manifest, &vectors).expect("the exported index must validate")
}

fn load_harness() -> Option<Harness> {
    let Some(model_dir) = resolve_model_dir() else {
        eprintln!(
            "Skipping: set {MODEL_DIR_ENV_VAR} to a model directory prepared by \
             AnimationModelTraining scripts/orchestrator_router/export_router_index.py"
        );
        return None;
    };

    let fixture: ParityFixture = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join(PARITY_FILENAME))
            .expect("router_parity.json must be readable"),
    )
    .expect("router_parity.json must parse");

    Some(Harness {
        encoder: SentenceEncoder::from_model_dir(&model_dir).expect("the encoder must load"),
        index: load_index(&model_dir),
        fixture,
    })
}

struct Disagreement {
    dataset: String,
    utterance: String,
    expected: String,
    actual: String,
    expected_score: f32,
    actual_score: f32,
}

/// Ranked with no threshold, because the comparison is of the route the encoder
/// and the tie-break arrive at. Rejection is a separate decision with its own
/// operating point, and folding it in here would hide a disagreement behind it.
fn decide(encoder: &mut SentenceEncoder, index: &ExemplarIndex, utterance: &str) -> (Route, f32) {
    let query = encoder
        .encode(utterance)
        .expect("the encoder must produce a vector");

    let decision = route_utterance(
        RoutingRequest {
            utterance,
            query_vector: &query,
            mode: OrchestratorMode::AllowEdit,
            raw_top_score: None,
        },
        index,
        RouterThresholds {
            tau_reject: f32::NEG_INFINITY,
            delta: 0.0,
            tau_confirm: 0.0,
            tau_raw: 0.0,
        },
    );

    match decision {
        RouterDecision::Accept { route, score, .. } => (route, score),
        other => panic!("an unthresholded decision must accept, got {other:?} for {utterance:?}"),
    }
}

fn collect_disagreements(harness: &mut Harness) -> Vec<Disagreement> {
    let Harness {
        encoder,
        index,
        fixture,
    } = harness;

    fixture
        .cases
        .iter()
        .filter_map(|case| {
            let (route, score) = decide(encoder, index, &case.utterance);
            let agrees = route.id() == case.route && (score - case.score).abs() <= SCORE_TOLERANCE;

            (!agrees).then(|| Disagreement {
                dataset: case.dataset.clone(),
                utterance: case.utterance.clone(),
                expected: case.route.clone(),
                actual: route.id(),
                expected_score: case.score,
                actual_score: score,
            })
        })
        .collect()
}

#[test]
fn the_rust_router_reaches_the_evaluated_decision_for_every_labelled_utterance() {
    let Some(mut harness) = load_harness() else {
        return;
    };
    assert_eq!(
        harness.fixture.stage_a, "polarity",
        "the fixture must be exported with the tie-break the engine runs"
    );
    assert!(!harness.fixture.cases.is_empty());

    // Re-exporting one file and not the other would otherwise pass: the decisions
    // would be replayed against an index they were never produced from.
    assert_eq!(
        harness.index.exemplar_count(),
        harness.fixture.exemplars,
        "the index and the parity fixture come from different exports"
    );

    let case_count = harness.fixture.cases.len();
    let disagreements = collect_disagreements(&mut harness);

    for item in &disagreements {
        eprintln!(
            "{} {:?}: expected {} at {:.5}, got {} at {:.5}",
            item.dataset,
            item.utterance,
            item.expected,
            item.expected_score,
            item.actual,
            item.actual_score
        );
    }
    assert!(
        disagreements.is_empty(),
        "{} of {case_count} utterances disagree with the evaluation driver",
        disagreements.len()
    );
}

/// The index is the file the engine trusts without re-deriving anything, so a
/// truncated or half-written export has to be caught on load rather than turned
/// into a ranking against whatever the trailing bytes were.
#[test]
fn a_truncated_export_of_the_real_index_is_refused() {
    let Some(model_dir) = resolve_model_dir() else {
        eprintln!("Skipping: {MODEL_DIR_ENV_VAR} is not set to a prepared model directory");
        return;
    };

    let manifest = std::fs::read_to_string(model_dir.join(MANIFEST_FILENAME))
        .expect("router_index.json must be readable");
    let mut vectors =
        std::fs::read(model_dir.join(VECTORS_FILENAME)).expect("router_index.f32 must be readable");
    vectors.truncate(vectors.len() - 4);

    assert!(ExemplarIndex::from_export(&manifest, &vectors).is_err());
}
