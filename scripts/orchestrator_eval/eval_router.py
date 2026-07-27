"""Evaluate the router against a labelled utterance set.

Routes and exemplars are the router's index; the evaluated utterances are not in
it. The escape-hatch cases carry no route and must fall below the rejection
threshold rather than being classified.

`--dataset heldout` is the number to quote. `--dataset devset` was the corpus the
retired keyword rules were written against, so it reads high for reasons that do
not carry over.

Run:
    .venv-orchestrator-eval/bin/python scripts/orchestrator_eval/eval_router.py \
        --embedder contextual --stage-a polarity --output results/router.json
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contextual_embedder import ContextualEmbedder
from dataset import DATASET_NAMES, DEFAULT_DATASET, load_dataset, read_jsonl
from local_paths import find_model_dir
from normalize import normalize_utterance
from polarity import load_tie_breaker
from route_schema import ESCAPE_ROUTE, build_routes, resolve_expected_route
from static_embedder import POOLING_STRATEGIES, StaticEmbedder

STATIC_MODEL_NAME = "gemma-3-270m-it-ONNX"
CONTEXTUAL_MODEL_NAME = "multilingual-e5-small"
RAW_ENCODER_DIR = "models/gemma/e5-raw"
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class Prediction:
    utterance: str
    lang: str
    expected_route: str
    predicted_route: str
    top_score: float
    margin: float
    raw_top_score: float = 0.0


STAGE_A_MODES = ("none", "polarity")


class StageA:
    """The deterministic stage that runs after the encoder has ranked the routes.

    It replaced a rule table that decided routes outright and so bypassed the
    rejection threshold; on held-out that table fired 47 times at 0.468 precision
    and cost 19 cases. This stage can only reorder the encoder's own top two, and
    only when those two are declared opposite poles of one axis, so it never
    introduces a route the encoder did not already rank second.
    """

    def __init__(self, mode: str):
        self.tie_breaker = load_tie_breaker() if mode == "polarity" else None

    def break_tie(
        self, utterance: str, ranked: list[tuple[str, float]]
    ) -> tuple[str, float, bool]:
        if self.tie_breaker is None or len(ranked) < 2:
            return ranked[0][0], ranked[0][1], False

        outcome = self.tie_breaker.resolve(normalize_utterance(utterance), ranked)
        winner_score = next(score for route, score in ranked if route == outcome.winner)
        return outcome.winner, winner_score, outcome.swapped


def load_labelled_utterances(dataset: str) -> list[dict]:
    rows = load_dataset(dataset)
    for row in rows:
        row["expected_route"] = resolve_expected_route(row["tool"], row["args"])
    return rows


def aggregate_route_scores(
    similarities: np.ndarray,
    route_slices: dict[str, slice],
    aggregation: str,
    allowed_routes: set[str] | None,
) -> dict[str, float]:
    scores = {}
    for route_id, span in route_slices.items():
        if allowed_routes is not None and route_id not in allowed_routes:
            continue

        block = similarities[span]
        if aggregation == "max":
            scores[route_id] = float(block.max())
        else:
            top = np.sort(block)[-3:]
            scores[route_id] = float(top.mean())
    return scores


class RouteHead:
    """The linear classifier SetFit fits on the adapted embeddings.

    Stored as plain coefficients so the evaluation venv needs numpy alone. Its
    softmax probability replaces the cosine score, which matters because the two
    are not comparable: cosine saturates near 1 for every utterance, while the
    probability is calibrated against the other routes.
    """

    def __init__(self, head_path: Path):
        payload = json.loads(head_path.read_text(encoding="utf-8"))
        self.classes = payload["classes"]
        self.coefficients = np.array(payload["coefficients"], dtype=np.float32)
        self.intercepts = np.array(payload["intercepts"], dtype=np.float32)

    def score_routes(self, utterance_vector: np.ndarray) -> dict[str, float]:
        logits = self.coefficients @ utterance_vector + self.intercepts
        exponentiated = np.exp(logits - logits.max())
        probabilities = exponentiated / exponentiated.sum()
        return dict(zip(self.classes, probabilities.tolist()))


def rank_scores(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def score_margin(ranked: list[tuple[str, float]]) -> float:
    if len(ranked) < 2:
        return ranked[0][1]
    return ranked[0][1] - ranked[1][1]


def rank_routes(
    utterance_vector: np.ndarray,
    exemplar_matrix: np.ndarray,
    route_slices: dict[str, slice],
    aggregation: str,
    allowed_routes: set[str] | None = None,
    head: "RouteHead | None" = None,
) -> list[tuple[str, float]]:
    if head is not None:
        scores = head.score_routes(utterance_vector)
        if allowed_routes is not None:
            scores = {route: score for route, score in scores.items() if route in allowed_routes}
        return rank_scores(scores)

    similarities = exemplar_matrix @ utterance_vector
    return rank_scores(
        aggregate_route_scores(similarities, route_slices, aggregation, allowed_routes)
    )


def predict_route(
    utterance_vector: np.ndarray,
    exemplar_matrix: np.ndarray,
    route_slices: dict[str, slice],
    aggregation: str,
    allowed_routes: set[str] | None = None,
    head: "RouteHead | None" = None,
) -> tuple[str, float, float]:
    ranked = rank_routes(
        utterance_vector, exemplar_matrix, route_slices, aggregation, allowed_routes, head
    )
    return ranked[0][0], ranked[0][1], score_margin(ranked)


def build_exemplar_index(
    embedder: StaticEmbedder, exemplars: list[dict], route_ids: list[str]
) -> tuple[np.ndarray, dict[str, slice]]:
    ordered = sorted(exemplars, key=lambda row: route_ids.index(row["route"]))
    matrix = embedder.encode_batch([row["utterance"] for row in ordered])

    slices: dict[str, slice] = {}
    start = 0
    for index, row in enumerate(ordered):
        if index + 1 == len(ordered) or ordered[index + 1]["route"] != row["route"]:
            slices[row["route"]] = slice(start, index + 1)
            start = index + 1
    return matrix, slices


def sweep_rejection_threshold(predictions: list[Prediction]) -> list[dict]:
    escape = [p for p in predictions if p.expected_route == ESCAPE_ROUTE]
    routed = [p for p in predictions if p.expected_route != ESCAPE_ROUTE]
    correct = [p for p in routed if p.predicted_route == p.expected_route]

    curve = []
    for threshold in sorted({round(p.top_score, 4) for p in predictions}):
        curve.append(
            {
                "threshold": threshold,
                "escape_recall": sum(p.top_score < threshold for p in escape) / len(escape) if len(escape) > 0 else 0.0,
                "retained": sum(p.top_score >= threshold for p in correct) / len(correct) if len(correct) > 0 else 1.0,
                "wrong_executed": sum(
                    p.top_score >= threshold
                    for p in routed
                    if p.predicted_route != p.expected_route
                )
                / len(routed) if len(routed) > 0 else 0.0,
            }
        )
    return curve


def summarize(predictions: list[Prediction], elapsed_seconds: float, tau1: float = 0.93, tau2: float = 0.90) -> dict:
    routed = [p for p in predictions if p.expected_route != ESCAPE_ROUTE]
    correct = [p for p in routed if p.predicted_route == p.expected_route]
    tool_correct = [
        p for p in routed if p.predicted_route.split(":")[0] == p.expected_route.split(":")[0]
    ]

    curve = sweep_rejection_threshold(predictions)
    full_escape = [point for point in curve if point["escape_recall"] >= 1.0]

    # Composite action point: accept if (setfit top1 >= tau1 AND raw top1 >= tau2)
    # Escape recall: fraction of escape cases where the condition is NOT met (rejected)
    escape = [p for p in predictions if p.expected_route == ESCAPE_ROUTE]
    composite_escape_recall = 0.0
    composite_retained = 1.0
    composite_route_accuracy = 0.0
    if escape:
        composite_escape_recall = sum(
            not (p.top_score >= tau1 and p.raw_top_score >= tau2) for p in escape
        ) / len(escape)
    # Retained: fraction of correct routed cases where the condition IS met (accepted)
    if correct:
        composite_retained = sum(
            p.top_score >= tau1 and p.raw_top_score >= tau2 for p in correct
        ) / len(correct)
    # Route accuracy: among accepted routed cases, what fraction are correct?
    accepted_routed = [p for p in routed if p.top_score >= tau1 and p.raw_top_score >= tau2]
    if accepted_routed:
        composite_route_accuracy = sum(
            p.predicted_route == p.expected_route for p in accepted_routed
        ) / len(accepted_routed)

    return {
        "case_count": len(predictions),
        "route_accuracy": len(correct) / len(routed) if len(routed) > 0 else 0.0,
        "tool_accuracy": len(tool_correct) / len(routed) if len(routed) > 0 else 0.0,
        "route_accuracy_en": _accuracy_for_language(routed, "en"),
        "route_accuracy_ja": _accuracy_for_language(routed, "ja"),
        "mean_top_score_correct": float(np.mean([p.top_score for p in correct])),
        "mean_top_score_escape": float(
            np.mean([p.top_score for p in predictions if p.expected_route == ESCAPE_ROUTE])
        ),
        "full_escape_operating_point": max(
            full_escape, key=lambda point: point["retained"], default=None
        ),
        "threshold_curve": curve,
        "seconds_per_utterance": elapsed_seconds / len(predictions) if len(predictions) > 0 else 0.0,
        "composite_escape_recall": composite_escape_recall,
        "composite_retained": composite_retained,
        "composite_route_accuracy": composite_route_accuracy,
    }


def _accuracy_for_language(routed: list[Prediction], lang: str) -> float:
    subset = [p for p in routed if p.lang == lang]
    if not subset:
        return 0.0
    return sum(p.predicted_route == p.expected_route for p in subset) / len(subset)


def resolve_model_dir(embedder_name: str, model_dir: str | None) -> str:
    if model_dir:
        return model_dir
    default_name = STATIC_MODEL_NAME if embedder_name == "static" else CONTEXTUAL_MODEL_NAME
    return find_model_dir(default_name)


def build_embedder(embedder_name: str, model_dir: str, pooling: str):
    if embedder_name == "static":
        return StaticEmbedder(model_dir, pooling=pooling)
    return ContextualEmbedder(model_dir)


def predict_all(
    cases: list[dict],
    utterance_vectors: np.ndarray,
    exemplar_matrix: np.ndarray,
    route_slices: dict[str, slice],
    aggregation: str,
    stage_a: "StageA",
    head: RouteHead | None = None,
    raw_exemplar_matrix: np.ndarray | None = None,
    raw_query_vectors: np.ndarray | None = None,
) -> tuple[list[Prediction], int]:
    predictions: list[Prediction] = []
    intervened = 0

    for i, (row, vector) in enumerate(zip(cases, utterance_vectors)):
        ranked = rank_routes(vector, exemplar_matrix, route_slices, aggregation, None, head)
        predicted, score, swapped = stage_a.break_tie(row["utterance"], ranked)
        intervened += swapped
        raw_score = 0.0
        if raw_exemplar_matrix is not None and raw_query_vectors is not None:
            raw_query_vector = raw_query_vectors[i]
            raw_score = float(np.max(raw_query_vector @ raw_exemplar_matrix.T))
        predictions.append(
            Prediction(
                row["utterance"],
                row["lang"],
                row["expected_route"],
                predicted,
                score,
                score_margin(ranked),
                raw_score,
            )
        )

    return predictions, intervened

def evaluate(
    embedder_name: str,
    model_dir: str | None,
    pooling: str,
    aggregation: str,
    stage_a_mode: str,
    dataset: str,
    scorer: str,
) -> dict:
    routes = build_routes()
    route_ids = [route.route_id for route in routes]
    exemplars = read_jsonl(SCRIPT_DIR / "exemplars.jsonl")
    cases = load_labelled_utterances(dataset)

    resolved_model_dir = resolve_model_dir(embedder_name, model_dir)
    embedder = build_embedder(embedder_name, resolved_model_dir, pooling)
    embedder.fit_corpus([row["utterance"] for row in exemplars])
    exemplar_matrix, route_slices = build_exemplar_index(embedder, exemplars, route_ids)
    stage_a = StageA(stage_a_mode)
    head = (
        RouteHead(Path(resolved_model_dir) / "route_head.json")
        if scorer == "route_head"
        else None
    )

    # Load raw exemplar matrix from pre-computed raw_index.f32 in RAW_ENCODER_DIR
    raw_exemplar_matrix: np.ndarray | None = None
    raw_query_vectors: np.ndarray | None = None
    raw_index_path = Path(resolved_model_dir) / "raw_index.f32"
    if raw_index_path.exists():
        raw_meta = json.loads((Path(resolved_model_dir) / "raw_index.json").read_text())
        raw_flat = np.fromfile(raw_index_path, dtype=np.float32)
        raw_exemplar_matrix = raw_flat.reshape(-1, raw_meta["dimensions"])
        # Instantiate raw embedder and encode query utterances
        raw_embedder = ContextualEmbedder(RAW_ENCODER_DIR)
        raw_query_vectors = raw_embedder.encode_batch([row["utterance"] for row in cases])

    start = time.perf_counter()
    utterance_vectors = embedder.encode_batch([row["utterance"] for row in cases])
    predictions, intervened = predict_all(
        cases, utterance_vectors, exemplar_matrix, route_slices, aggregation, stage_a, head,
        raw_exemplar_matrix=raw_exemplar_matrix,
        raw_query_vectors=raw_query_vectors,
    )
    elapsed = time.perf_counter() - start

    result = summarize(predictions, elapsed)
    result["dataset"] = dataset
    result["scorer"] = scorer
    result["embedder"] = embedder_name
    result["pooling"] = pooling if embedder_name == "static" else "n/a"
    result["aggregation"] = aggregation
    result["stage_a"] = stage_a_mode
    result["stage_a_intervened"] = intervened
    result["route_count"] = len(routes)
    result["exemplar_count"] = len(exemplars)
    result["predictions"] = [vars(p) for p in predictions]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", default="static", choices=("static", "contextual"))
    parser.add_argument("--model-dir")
    parser.add_argument("--pooling", default="unit_mean", choices=POOLING_STRATEGIES)
    parser.add_argument("--aggregation", default="max", choices=("max", "mean_top3"))
    parser.add_argument("--stage-a", default="polarity", choices=STAGE_A_MODES)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=DATASET_NAMES)
    parser.add_argument("--scorer", default="exemplar_max", choices=("exemplar_max", "route_head"))
    parser.add_argument("--output")
    parser.add_argument("--tau-reject", type=float, default=0.93, help="Setfit threshold for composite action point")
    parser.add_argument("--tau-raw", type=float, default=0.90, help="Raw encoder threshold for composite action point")
    arguments = parser.parse_args()

    result = evaluate(
        arguments.embedder,
        arguments.model_dir,
        arguments.pooling,
        arguments.aggregation,
        arguments.stage_a,
        arguments.dataset,
        arguments.scorer,
    )

    print(
        f"dataset={result['dataset']} scorer={result['scorer']} "
        f"embedder={result['embedder']} aggregation={result['aggregation']} "
        f"stage_a={result['stage_a']} "
        f"(intervened {result['stage_a_intervened']}/{result['case_count']})"
    )
    print(f"  route accuracy   {result['route_accuracy']:.3f}")
    print(f"  tool accuracy    {result['tool_accuracy']:.3f}")
    print(
        f"  en / ja          {result['route_accuracy_en']:.3f}"
        f" / {result['route_accuracy_ja']:.3f}"
    )
    print(f"  mean score corr  {result['mean_top_score_correct']:.3f}")
    print(f"  mean score esc   {result['mean_top_score_escape']:.3f}")
    print(f"  escape 100% pt   {result['full_escape_operating_point']}")
    print(f"  sec/utterance    {result['seconds_per_utterance']:.5f}")
    print(
        f"  composite (tau1={arguments.tau_reject}, tau2={arguments.tau_raw}): "
        f"escape_recall={result['composite_escape_recall']:.3f} "
        f"retained={result['composite_retained']:.3f} "
        f"route_accuracy={result['composite_route_accuracy']:.3f}"
    )

    if arguments.output:
        output_path = Path(arguments.output)
        if not output_path.is_absolute():
            output_path = SCRIPT_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
