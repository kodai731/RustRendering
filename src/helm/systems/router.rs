//! Stage B: ranks the routes by how close the utterance is to their exemplars.
//!
//! The route a small model can pick reliably is one with no arguments left to
//! fill, which is why a route is a tool plus the enum arguments that appear in the
//! utterance. Ranking is then a cosine comparison against per-route exemplars and
//! nothing else: no generation, no parsing, no prompt. `AnimationModelTraining scripts/helm_router/`
//! is where the accuracy of that claim is measured, and this file must rank the
//! same way or the measurement stops describing the engine — per-route score is
//! the single best exemplar, ties keep the route table's order, and the score the
//! threshold sees is the winner's, including after the polarity tie-break has
//! promoted a runner-up.
//!
//! Deliberately free of both the encoder and the filesystem. The vectors arrive
//! already computed, which keeps this layer pure enough to test the ranking on
//! hand-written vectors and leaves the ONNX session to
//! `thyllore_ml_core::sentence_encoder`.

use std::ops::Range;

use serde::Deserialize;
use thiserror::Error;

use crate::helm::components::route::{HelmMode, Route};
use crate::helm::systems::normalize::normalize_utterance;
use crate::helm::systems::polarity_tiebreak::{break_polarity_tie, share_axis, TieBreakOutcome};

/// Chosen on `devset` as the lowest threshold that executes no wrong route there:
/// it rejects 7 of 9 escape utterances and keeps 62 of 63 correct routes. Raising
/// it to reject all 9 costs 60 points of retention, so full escape recall is not
/// the operating point.
///
/// The criterion does not transfer intact. On `heldout` this same threshold keeps
/// 80 of 98 correct routes but lets 9 of 116 wrong routes through, where `devset`
/// promised none — top-1 score alone is a weak confidence signal, and that
/// remains open rather than being papered over by tuning against `heldout`.
pub const DEFAULT_REJECTION_THRESHOLD: f32 = 0.825;

/// Calibrated thresholds from retained_improvement P2 (held-out retained 0.790 / escape recall 0.833 / route 0.862; delta from devset Clarify sweep).
pub const TUNED_TAU_REJECT: f32 = 0.93;
pub const TUNED_TAU_RAW: f32 = 0.90;
pub const TUNED_DELTA: f32 = 0.0025;
pub const TUNED_TAU_CONFIRM: f32 = 0.95;
const UNIT_LENGTH_TOLERANCE: f32 = 1e-3;
const BYTES_PER_SCALAR: usize = 4;

#[derive(Debug, Error, PartialEq)]
pub enum IndexError {
    #[error("router index manifest is not readable: {0}")]
    UnreadableManifest(String),
    #[error("router index declares {0} embedding dimensions")]
    EmptyEmbedding(usize),
    #[error("router index names route {0}, which this build does not have")]
    UnknownRoute(String),
    #[error("router index gives route {0} no exemplars")]
    RouteWithoutExemplars(String),
    #[error("router index vectors are {found} bytes, expected {expected}")]
    VectorSizeMismatch { expected: usize, found: usize },
    #[error("router index row {row} has length {length}, expected unit length")]
    RowNotUnitLength { row: usize, length: f32 },
}

#[derive(Deserialize)]
struct Manifest {
    dimensions: usize,
    #[serde(default)]
    exemplars_sha256: Option<String>,
    routes: Vec<ManifestRoute>,
}

#[derive(Deserialize)]
struct ManifestRoute {
    route: String,
    exemplars: usize,
}

#[derive(Debug)]
struct RouteBlock {
    route: Route,
    rows: Range<usize>,
}

/// The exemplar vectors the router compares against, one contiguous block per route.
///
/// Built from an export rather than by encoding the exemplars at startup: 464
/// forward passes cost seconds the editor does not have, and an export also pins
/// the index to the encoder it was measured with.
#[derive(Debug)]
pub struct ExemplarIndex {
    dimensions: usize,
    exemplars_sha256: Option<String>,
    blocks: Vec<RouteBlock>,
    vectors: Vec<f32>,
}

impl ExemplarIndex {
    /// Validates the export at the boundary so ranking can trust every row.
    ///
    /// `vector_bytes` is little-endian f32, rows in manifest order. A partial or
    /// mismatched export is the realistic failure — a stale file against a
    /// retrained encoder — and it has to fail here rather than produce a ranking
    /// that looks plausible.
    pub fn from_export(manifest_json: &str, vector_bytes: &[u8]) -> Result<Self, IndexError> {
        let manifest: Manifest = serde_json::from_str(manifest_json)
            .map_err(|error| IndexError::UnreadableManifest(error.to_string()))?;

        if manifest.dimensions == 0 {
            return Err(IndexError::EmptyEmbedding(manifest.dimensions));
        }

        let blocks = build_blocks(&manifest.routes)?;
        let rows = blocks.last().map_or(0, |block| block.rows.end);
        let expected = rows * manifest.dimensions * BYTES_PER_SCALAR;
        if vector_bytes.len() != expected {
            return Err(IndexError::VectorSizeMismatch {
                expected,
                found: vector_bytes.len(),
            });
        }

        let index = Self {
            dimensions: manifest.dimensions,
            exemplars_sha256: manifest.exemplars_sha256,
            blocks,
            vectors: decode_vectors(vector_bytes),
        };
        index.verify_rows_are_unit_length()?;
        Ok(index)
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn exemplar_count(&self) -> usize {
        self.vectors.len() / self.dimensions
    }

    pub fn exemplars_sha256(&self) -> Option<&str> {
        self.exemplars_sha256.as_deref()
    }

    fn row(&self, row: usize) -> &[f32] {
        &self.vectors[row * self.dimensions..(row + 1) * self.dimensions]
    }

    fn verify_rows_are_unit_length(&self) -> Result<(), IndexError> {
        for row in 0..self.exemplar_count() {
            let length = self
                .row(row)
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            if (length - 1.0).abs() > UNIT_LENGTH_TOLERANCE {
                return Err(IndexError::RowNotUnitLength { row, length });
            }
        }
        Ok(())
    }

    fn best_similarity(&self, block: &RouteBlock, query: &[f32]) -> f32 {
        block
            .rows
            .clone()
            .map(|row| dot(self.row(row), query))
            .fold(f32::NEG_INFINITY, f32::max)
    }
}

fn build_blocks(routes: &[ManifestRoute]) -> Result<Vec<RouteBlock>, IndexError> {
    let mut blocks = Vec::with_capacity(routes.len());
    let mut start = 0;

    for entry in routes {
        let route = Route::from_id(&entry.route)
            .ok_or_else(|| IndexError::UnknownRoute(entry.route.clone()))?;
        if entry.exemplars == 0 {
            return Err(IndexError::RouteWithoutExemplars(entry.route.clone()));
        }

        blocks.push(RouteBlock {
            route,
            rows: start..start + entry.exemplars,
        });
        start += entry.exemplars;
    }
    Ok(blocks)
}

fn decode_vectors(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(BYTES_PER_SCALAR)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// Every route the mode allows, best score first.
///
/// The sort is stable and the blocks are in route-table order, so two routes with
/// the same score come back in that order — the evaluation driver's Python sort
/// behaves identically, and the tie-break downstream reads position, not score.
pub fn rank_routes(
    index: &ExemplarIndex,
    query: &[f32],
    mode: HelmMode,
) -> Vec<(Route, f32)> {
    assert_eq!(
        query.len(),
        index.dimensions(),
        "the encoder and the router index must agree on embedding width"
    );

    let mut ranked: Vec<(Route, f32)> = index
        .blocks
        .iter()
        .filter(|block| block.route.is_available_in(mode))
        .map(|block| (block.route, index.best_similarity(block, query)))
        .collect();

    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    ranked
}

pub struct RoutingRequest<'a> {
    pub utterance: &'a str,
    pub query_vector: &'a [f32],
    pub mode: HelmMode,
    pub raw_top_score: Option<f32>,
}

/// What the router concluded. Rejection is a distinct outcome rather than a route
/// with a low score, so a caller cannot dispatch one by forgetting to compare.
#[derive(Clone, Debug, PartialEq)]
pub enum RouterDecision {
    Accept {
        route: Route,
        score: f32,
        needs_confirm: bool,
        raw_near_miss: bool,
    },
   /// Below the threshold. `best` is what the encoder preferred, which is what a
    Reject {
        best: Route,
        score: f32,
    },
    Clarify {
        candidates: Vec<(Route, f32)>,
    },
    /// The mode left no route to choose from.
    NoCandidate,
}

#[derive(Clone, Copy, Debug)]
pub struct RouterThresholds {
    pub tau_reject: f32,
    pub delta: f32,
    pub tau_confirm: f32,
    pub tau_raw: f32,
    pub tau_raw_nearmiss: f32,
}

/// Route an utterance by ranking against the exemplar index.
///
/// 昇格側・縮退survivorは自身のスコアで閾値判定する (ゲート迂回不可)
pub fn route_utterance(
    request: RoutingRequest<'_>,
    index: &ExemplarIndex,
    thresholds: RouterThresholds,
) -> RouterDecision {
    let ranked = rank_routes(index, request.query_vector, request.mode);

    // If top-1 is EscapeAnchor, reject immediately — bypass threshold gate and polarity tie-break.
    if let Some((Route::EscapeAnchor, score)) = ranked.first() {
        return RouterDecision::Reject {
            best: Route::EscapeAnchor,
            score: *score,
        };
    }

   let raw_near_miss = if let Some(raw) = request.raw_top_score {
        if raw < thresholds.tau_raw_nearmiss {
            return match ranked.first() {
                Some((route, score)) => RouterDecision::Reject {
                    best: *route,
                    score: *score,
                },
                None => RouterDecision::NoCandidate,
            };
        }
        raw < thresholds.tau_raw
    } else {
        false
    };
    let normalized = normalize_utterance(request.utterance);

    let Some(outcome) = break_polarity_tie(&normalized, &ranked) else {
        return RouterDecision::NoCandidate;
    };

    fn score_of(ranked: &[(Route, f32)], route: Route) -> f32 {
        ranked.iter().find(|(r, _)| *r == route).map_or(f32::NEG_INFINITY, |(_, s)| *s)
    }

    let (winner, score) = match outcome {
        TieBreakOutcome::Decided(r) => (r, score_of(&ranked, r)),
        TieBreakOutcome::NotApplicable(r) => {
            if thresholds.delta > 0.0 && ranked.len() >= 2 && ranked[0].1 - ranked[1].1 < thresholds.delta {
                let candidates: Vec<(Route, f32)> = ranked.iter().take(3)
                    .filter(|(_, s)| *s >= thresholds.tau_reject).copied().collect();
                match candidates[..] {
                    [] => return RouterDecision::Reject { best: ranked[0].0, score: ranked[0].1 },
                    [single] => single,
                    _ => return RouterDecision::Clarify { candidates },
                }
            } else {
                (r, score_of(&ranked, r))
            }
        }
        TieBreakOutcome::Undecided(r) => {
            if thresholds.delta > 0.0 && ranked.len() >= 2 {
                let candidates: Vec<(Route, f32)> = ranked.iter().take(2)
                    .filter(|(_, s)| *s >= thresholds.tau_reject).copied().collect();
                match candidates[..] {
                    [] => return RouterDecision::Reject { best: ranked[0].0, score: ranked[0].1 },
                    [single] => single,
                    _ => return RouterDecision::Clarify { candidates },
                }
            } else {
                (r, score_of(&ranked, r))
            }
        }
    };

   if score >= thresholds.tau_reject {
        let needs_confirm = score < thresholds.tau_confirm;
        RouterDecision::Accept {
            route: winner,
            score,
            needs_confirm,
            raw_near_miss,
        }
   } else {
        RouterDecision::Reject {
            best: winner,
            score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helm::components::tool_call::{SeekPosition, VisibilityState};

    const DIMENSIONS: usize = 4;

    fn unit(values: [f32; DIMENSIONS]) -> [f32; DIMENSIONS] {
        let length = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        values.map(|value| value / length)
    }

    fn export(rows: &[(&str, Vec<[f32; DIMENSIONS]>)]) -> (String, Vec<u8>) {
        let routes: Vec<String> = rows
            .iter()
            .map(|(route, vectors)| {
                format!(r#"{{"route":"{route}","exemplars":{}}}"#, vectors.len())
            })
            .collect();
        let manifest = format!(
            r#"{{"dimensions":{DIMENSIONS},"routes":[{}]}}"#,
            routes.join(",")
        );

        let bytes = rows
            .iter()
            .flat_map(|(_, vectors)| vectors.iter())
            .flat_map(|vector| vector.iter().flat_map(|value| value.to_le_bytes()))
            .collect();
        (manifest, bytes)
    }

    fn load(rows: &[(&str, Vec<[f32; DIMENSIONS]>)]) -> ExemplarIndex {
        let (manifest, bytes) = export(rows);
        ExemplarIndex::from_export(&manifest, &bytes).expect("the fixture export is valid")
    }

    fn two_route_index() -> ExemplarIndex {
        load(&[
            ("play_animation", vec![unit([1.0, 0.0, 0.0, 0.0])]),
            (
                "pause_animation",
                vec![unit([0.0, 1.0, 0.0, 0.0]), unit([0.0, 0.9, 0.1, 0.0])],
            ),
        ])
    }

    #[test]
    fn a_route_scores_as_its_single_closest_exemplar() {
        let index = two_route_index();
        let ranked = rank_routes(
            &index,
            &unit([0.0, 0.9, 0.1, 0.0]),
            HelmMode::AllowEdit,
        );

        assert_eq!(ranked[0].0, Route::PauseAnimation);
        assert!(
            (ranked[0].1 - 1.0).abs() < 1e-5,
            "the exact exemplar should score 1.0, got {}",
            ranked[0].1
        );
    }

    /// Averaging would let a route with many mediocre exemplars outrank a route
    /// with one exact match, which is the opposite of what an index of paraphrases
    /// should do.
    #[test]
    fn extra_distant_exemplars_do_not_dilute_a_route() {
        let sharp = load(&[
            ("play_animation", vec![unit([1.0, 0.0, 0.0, 0.0])]),
            ("pause_animation", vec![unit([0.9, 0.1, 0.0, 0.0])]),
        ]);
        let padded = load(&[
            (
                "play_animation",
                vec![
                    unit([1.0, 0.0, 0.0, 0.0]),
                    unit([0.0, 0.0, 1.0, 0.0]),
                    unit([0.0, 0.0, 0.0, 1.0]),
                ],
            ),
            ("pause_animation", vec![unit([0.9, 0.1, 0.0, 0.0])]),
        ]);

        let query = unit([1.0, 0.0, 0.0, 0.0]);
        for index in [sharp, padded] {
            assert_eq!(
                rank_routes(&index, &query, HelmMode::AllowEdit)[0].0,
                Route::PlayAnimation
            );
        }
    }

    #[test]
    fn equal_scores_keep_the_route_table_order() {
        let index = load(&[
            ("pause_animation", vec![unit([1.0, 0.0, 0.0, 0.0])]),
            ("play_animation", vec![unit([1.0, 0.0, 0.0, 0.0])]),
        ]);
        let ranked = rank_routes(
            &index,
            &unit([1.0, 0.0, 0.0, 0.0]),
            HelmMode::AllowEdit,
        );

        assert_eq!(ranked[0].0, Route::PauseAnimation);
        assert_eq!(ranked[1].0, Route::PlayAnimation);
    }

    #[test]
    fn read_only_mode_never_ranks_an_edit_route() {
        let index = load(&[
            ("play_animation", vec![unit([1.0, 0.0, 0.0, 0.0])]),
            ("list_objects", vec![unit([0.0, 1.0, 0.0, 0.0])]),
        ]);
        let ranked = rank_routes(
            &index,
            &unit([1.0, 0.0, 0.0, 0.0]),
            HelmMode::ReadOnly,
        );

        assert_eq!(ranked, vec![(Route::ListObjects, 0.0)]);
    }

    #[test]
    fn a_mode_that_allows_nothing_yields_no_candidate() {
        let index = load(&[("play_animation", vec![unit([1.0, 0.0, 0.0, 0.0])])]);
       let decision = route_utterance(
            RoutingRequest {
                utterance: "play it",
                query_vector: &unit([1.0, 0.0, 0.0, 0.0]),
                mode: HelmMode::ReadOnly,
                raw_top_score: None,
            },
            &index,
            RouterThresholds {
                tau_reject: DEFAULT_REJECTION_THRESHOLD,
                delta: 0.0,
                tau_confirm: 0.0,
                tau_raw: 0.0,
                tau_raw_nearmiss: 0.0,
            },
        );

        assert_eq!(decision, RouterDecision::NoCandidate);
    }

    #[test]
    fn a_score_below_the_threshold_is_rejected_rather_than_dispatched() {
        let index = two_route_index();
        let decision = route_utterance(
            RoutingRequest {
                utterance: "what is the weather",
                query_vector: &unit([0.0, 0.0, 1.0, 0.0]),
                mode: HelmMode::AllowEdit,
                raw_top_score: None,
            },
            &index,
            RouterThresholds {
                tau_reject: DEFAULT_REJECTION_THRESHOLD,
                delta: 0.0,
                tau_confirm: 0.0,
                tau_raw: 0.0,
                tau_raw_nearmiss: 0.0,
            },
        );

        assert!(
            matches!(decision, RouterDecision::Reject { .. }),
            "got {decision:?}"
        );
    }

   /// The threshold must see the promoted route's own score. Reading the leader's
    /// instead would let a swap smuggle a route past a threshold it never met.
    #[test]
    fn a_promoted_runner_up_is_thresholded_on_its_own_score() {
        let index = load(&[
            ("seek_time:next_key", vec![unit([1.0, 0.0, 0.0, 0.0])]),
            ("seek_time:prev_key", vec![unit([0.0, 1.0, 0.0, 0.0])]),
        ]);
        let query = unit([0.995, 0.1, 0.0, 0.0]);
        let request = RoutingRequest {
            utterance: "hop to the key before this one",
            query_vector: &query,
            mode: HelmMode::AllowEdit,
            raw_top_score: None,
        };
        let ranked = rank_routes(&index, &query, HelmMode::AllowEdit);
        let runner_up_score = ranked[1].1;

       assert_eq!(ranked[0].0, Route::SeekTime(SeekPosition::NextKey));
        assert_eq!(
            route_utterance(request, &index, RouterThresholds { tau_reject: 0.0, delta: 0.0, tau_confirm: 0.0, tau_raw: 0.0, tau_raw_nearmiss: 0.0 }),
            RouterDecision::Accept {
                route: Route::SeekTime(SeekPosition::PrevKey),
                score: runner_up_score,
                needs_confirm: false,
                raw_near_miss: false,
            }
        );
    }
    #[test]
    fn a_promoted_runner_up_below_the_threshold_is_still_rejected() {
        let index = load(&[
            (
                "set_object_visibility:show",
                vec![unit([1.0, 0.0, 0.0, 0.0])],
            ),
            (
                "set_object_visibility:hide",
                vec![unit([0.0, 1.0, 0.0, 0.0])],
            ),
        ]);
        let query = unit([0.9, 0.3, 0.0, 0.0]);

        assert_eq!(
            route_utterance(
                RoutingRequest {
                    utterance: "make the cube invisible",
                    query_vector: &query,
                    mode: HelmMode::AllowEdit,
                    raw_top_score: None,
                },
                &index,
                RouterThresholds {
                    tau_reject: DEFAULT_REJECTION_THRESHOLD,
                    delta: 0.0,
                    tau_confirm: 0.0,
                    tau_raw: 0.0,
                    tau_raw_nearmiss: 0.0,
                },
            ),
            RouterDecision::Reject {
                best: Route::SetObjectVisibility(VisibilityState::Hide),
                score: rank_routes(&index, &query, HelmMode::AllowEdit)[1].1,
            }
        );
    }

    #[test]
    fn an_export_naming_an_unknown_route_is_refused() {
        let manifest = r#"{"dimensions":4,"routes":[{"route":"teleport","exemplars":1}]}"#;
        assert_eq!(
            ExemplarIndex::from_export(manifest, &[0; 16]).unwrap_err(),
            IndexError::UnknownRoute("teleport".to_string())
        );
    }

    #[test]
    fn an_export_with_a_route_that_has_no_exemplars_is_refused() {
        let manifest = r#"{"dimensions":4,"routes":[{"route":"undo","exemplars":0}]}"#;
        assert_eq!(
            ExemplarIndex::from_export(manifest, &[]).unwrap_err(),
            IndexError::RouteWithoutExemplars("undo".to_string())
        );
    }

    /// A truncated export is the realistic accident, and it would otherwise leave
    /// the last routes ranking against whatever the trailing rows happened to be.
    #[test]
    fn a_truncated_export_is_refused() {
        let (manifest, mut bytes) = export(&[
            ("undo", vec![unit([1.0, 0.0, 0.0, 0.0])]),
            ("redo", vec![unit([0.0, 1.0, 0.0, 0.0])]),
        ]);
        bytes.truncate(bytes.len() - BYTES_PER_SCALAR);

        assert_eq!(
            ExemplarIndex::from_export(&manifest, &bytes).unwrap_err(),
            IndexError::VectorSizeMismatch {
                expected: 32,
                found: 28,
            }
        );
    }

    /// Cosine is a dot product only while both sides are unit length. An export
    /// that skipped normalization would silently rank by magnitude.
    #[test]
    fn an_export_that_skipped_normalization_is_refused() {
        let (manifest, bytes) = export(&[("undo", vec![[2.0, 0.0, 0.0, 0.0]])]);
        assert_eq!(
            ExemplarIndex::from_export(&manifest, &bytes).unwrap_err(),
            IndexError::RowNotUnitLength {
                row: 0,
                length: 2.0,
            }
        );
    }

    #[test]
    fn an_unparseable_manifest_is_refused() {
        assert!(matches!(
            ExemplarIndex::from_export("not json", &[]),
            Err(IndexError::UnreadableManifest(_))
        ));
    }

    /// Undecided axis pair where only 1 is >= tau_reject -> Accept (the surviving one).
    #[test]
    fn undecided_axis_pair_only_one_above_tau_reject_is_accepted() {
        let index = load(&[
            ("seek_time:start", vec![unit([1.0, 0.0, 0.0, 0.0])]),
            ("seek_time:end", vec![unit([0.0, 1.0, 0.0, 0.0])]),
        ]);
        // Query very close to start, far from end
        let query = unit([0.995, 0.1, 0.0, 0.0]);

        let decision = route_utterance(
            RoutingRequest {
                utterance: "go to the last",
                query_vector: &query,
                mode: HelmMode::AllowEdit,
                raw_top_score: None,
            },
            &index,
            RouterThresholds {
                tau_reject: 0.95,
                delta: 0.1,
                tau_confirm: 0.0,
                tau_raw: 0.0,
                tau_raw_nearmiss: 0.0,
            },
        );

        // start score = 0.995*1.0 + 0.1*0.0 = 0.995
        // end score = 0.995*0.0 + 0.1*1.0 = 0.1
        // ranked: [start(0.995), end(0.1)]
        // "last" -> polarity swap to end, so Undecided(end)
        // axis pair candidates from top 2: [start, end] (both share_axis with end)
        // filtered by tau_reject=0.95: start(0.995)>=0.95 yes, end(0.1)>=0.95 no -> 1 candidate
        // single survivor = start -> Accept with start's score

        match decision {
          RouterDecision::Accept { route, score, needs_confirm, .. } => {
                assert_eq!(route, Route::SeekTime(SeekPosition::Start));
                assert!((score - 0.995).abs() < 1e-4, "score = {}", score);
                assert!(!needs_confirm);
            }
            other => panic!("expected Accept, got {:?}", other),
        }
    }

    /// Undecided axis pair where both are < tau_reject -> Reject.
    #[test]
    fn undecided_axis_pair_both_below_tau_reject_is_rejected() {
        let index = load(&[
            ("seek_time:start", vec![unit([1.0, 0.0, 0.0, 0.0])]),
            ("seek_time:end", vec![unit([0.98, 0.1, 0.0, 0.0])]),
        ]);
        // Query far from both
        let query = unit([0.5, 0.5, 0.5, 0.5]);

        let decision = route_utterance(
            RoutingRequest {
                utterance: "go to the last",
                query_vector: &query,
                mode: HelmMode::AllowEdit,
                raw_top_score: None,
            },
            &index,
            RouterThresholds {
                tau_reject: 0.95,
                delta: 0.1,
                tau_confirm: 0.0,
                tau_raw: 0.0,
                tau_raw_nearmiss: 0.0,
            },
        );

        assert!(
            matches!(decision, RouterDecision::Reject { .. }),
            "expected Reject, got {:?}",
            decision
        );
    }

    /// Undecided (same axis, no polarity word) with large score difference (> delta) still returns Clarify.
    #[test]
    fn undecided_large_score_difference_still_clarifies() {
        let index = load(&[
            ("seek_time:start", vec![unit([1.0, 0.0, 0.0, 0.0])]),
            // [0.7, sqrt(1-0.49), 0, 0] = [0.7, 0.714, 0, 0]
            ("seek_time:end", vec![unit([0.7, 0.714, 0.0, 0.0])]),
        ]);
        // Query very close to seek_time:start
        let query = unit([1.0, 0.0, 0.0, 0.0]);

        let decision = route_utterance(
            RoutingRequest {
                utterance: "seek",
                query_vector: &query,
                mode: HelmMode::AllowEdit,
                raw_top_score: None,
            },
            &index,
            RouterThresholds {
                tau_reject: 0.5,
                delta: 0.1,
                tau_confirm: 0.8,
                tau_raw: 0.0,
                tau_raw_nearmiss: 0.0,
            },
        );

        // seek_time:start score = 1.0, seek_time:end score ~ 0.7
        // difference = 0.3 > delta(0.1), but Undecided is same-axis antonym with no polarity word
        // so it should still return Clarify with 2 candidates (both >= tau_reject=0.5)

        match decision {
            RouterDecision::Clarify { candidates } => {
                assert_eq!(candidates.len(), 2, "expected 2 candidates, got {:?}", candidates);
                assert_eq!(candidates[0].0, Route::SeekTime(SeekPosition::Start));
                assert!((candidates[0].1 - 1.0).abs() < 1e-6, "expected score ~1.0, got {}", candidates[0].1);
                assert_eq!(candidates[1].0, Route::SeekTime(SeekPosition::End));
                assert!((candidates[1].1 - 0.7).abs() < 1e-2, "expected score ~0.7, got {}", candidates[1].1);
            }
            other => panic!("expected Clarify, got {:?}", other),
        }
    }

    /// NotApplicable margin where only top1 is >= tau_reject -> Accept (not Clarify).
    #[test]
    fn not_applicable_margin_only_top1_above_tau_reject_is_accepted() {
        let index = load(&[
            ("play_animation", vec![unit([1.0, 0.0, 0.0, 0.0])]),
            // exemplar [0.92, 0.387, 0.0, 0.0] normalized: length = sqrt(0.92^2 + 0.387^2) = sqrt(0.8464 + 0.1498) = sqrt(0.9962) ~ 0.998
            // Actually let me compute: [0.92, sqrt(1-0.92^2), 0, 0] = [0.92, 0.392, 0, 0]
            ("pause_animation", vec![unit([0.92, 0.392, 0.0, 0.0])]),
        ]);
        // Query very close to play_animation
        let query = unit([1.0, 0.0, 0.0, 0.0]);

      let decision = route_utterance(
            RoutingRequest {
                utterance: "play it",
                query_vector: &query,
                mode: HelmMode::AllowEdit,
                raw_top_score: None,
            },
            &index,
            RouterThresholds {
                tau_reject: 0.95,
                delta: 0.1,
                tau_confirm: 0.0,
                tau_raw: 0.0,
                tau_raw_nearmiss: 0.0,
            },
        );
        // play_animation score = 1.0, pause_animation score ~ 0.92
        // margin = 1.0 - 0.92 = 0.08 < delta(0.1) -> margin path triggers
        // candidates from top 3 filtered by tau_reject=0.95: play(1.0)>=0.95 yes, pause(~0.92)>=0.95 no
        // 1 candidate -> single survivor = play -> Accept

        assert_eq!(
            decision,
            RouterDecision::Accept {
                route: Route::PlayAnimation,
                score: 1.0,
                needs_confirm: false,
                raw_near_miss: false,
            }
        );
    }
}
