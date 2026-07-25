"""Evaluate the embedding router against the 74-case tool-selection testset.

Routes and exemplars are the router's index; the testset is held out. The
escape-hatch cases have no route and must fall below the rejection threshold
rather than being classified.

Run:
    .venv-orchestrator-eval/bin/python scripts/orchestrator_eval/eval_router.py \
        --pooling unit_mean --output results/router_static_unit_mean.json
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contextual_embedder import ContextualEmbedder
from route_schema import ESCAPE_ROUTE, build_routes, resolve_expected_route
from static_embedder import POOLING_STRATEGIES, StaticEmbedder

STATIC_MODEL_DIR = "/home/kodai/Projects/CodeAgent/models/gemma-3-270m-it-ONNX"
CONTEXTUAL_MODEL_DIR = "/home/kodai/Projects/CodeAgent/models/multilingual-e5-small"
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class Prediction:
    utterance: str
    lang: str
    expected_route: str
    predicted_route: str
    top_score: float
    margin: float


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_testset(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    for row in rows:
        row["expected_route"] = resolve_expected_route(row["tool"], row["args"])
    return rows


def aggregate_route_scores(
    similarities: np.ndarray, route_slices: dict[str, slice], aggregation: str
) -> dict[str, float]:
    scores = {}
    for route_id, span in route_slices.items():
        block = similarities[span]
        if aggregation == "max":
            scores[route_id] = float(block.max())
        else:
            top = np.sort(block)[-3:]
            scores[route_id] = float(top.mean())
    return scores


def predict_route(
    utterance_vector: np.ndarray,
    exemplar_matrix: np.ndarray,
    route_slices: dict[str, slice],
    aggregation: str,
) -> tuple[str, float, float]:
    similarities = exemplar_matrix @ utterance_vector
    scores = aggregate_route_scores(similarities, route_slices, aggregation)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1]
    return ranked[0][0], ranked[0][1], margin


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

    candidates = sorted({round(p.top_score, 4) for p in predictions})
    curve = []
    for threshold in candidates:
        curve.append(
            {
                "threshold": threshold,
                "escape_recall": sum(p.top_score < threshold for p in escape) / len(escape),
                "retained": sum(p.top_score >= threshold for p in correct) / len(correct),
                "wrong_executed": sum(
                    p.top_score >= threshold
                    for p in routed
                    if p.predicted_route != p.expected_route
                )
                / len(routed),
            }
        )
    return curve


def summarize(predictions: list[Prediction], elapsed_seconds: float) -> dict:
    routed = [p for p in predictions if p.expected_route != ESCAPE_ROUTE]
    correct = [p for p in routed if p.predicted_route == p.expected_route]
    tool_correct = [
        p for p in routed if p.predicted_route.split(":")[0] == p.expected_route.split(":")[0]
    ]
    curve = sweep_rejection_threshold(predictions)
    full_escape = [point for point in curve if point["escape_recall"] >= 1.0]

    return {
        "case_count": len(predictions),
        "route_accuracy": len(correct) / len(routed),
        "tool_accuracy": len(tool_correct) / len(routed),
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
        "seconds_per_utterance": elapsed_seconds / len(predictions),
    }


def _accuracy_for_language(routed: list[Prediction], lang: str) -> float:
    subset = [p for p in routed if p.lang == lang]
    if not subset:
        return 0.0
    return sum(p.predicted_route == p.expected_route for p in subset) / len(subset)


def build_embedder(embedder_name: str, model_dir: str | None, pooling: str):
    if embedder_name == "static":
        return StaticEmbedder(model_dir or STATIC_MODEL_DIR, pooling=pooling)
    return ContextualEmbedder(model_dir or CONTEXTUAL_MODEL_DIR)


def evaluate(embedder_name: str, model_dir: str | None, pooling: str, aggregation: str) -> dict:
    routes = build_routes()
    route_ids = [route.route_id for route in routes]
    exemplars = read_jsonl(SCRIPT_DIR / "exemplars.jsonl")
    testset = load_testset(SCRIPT_DIR / "testset.jsonl")

    embedder = build_embedder(embedder_name, model_dir, pooling)
    embedder.fit_corpus([row["utterance"] for row in exemplars])
    exemplar_matrix, route_slices = build_exemplar_index(embedder, exemplars, route_ids)

    start = time.perf_counter()
    utterance_vectors = embedder.encode_batch([row["utterance"] for row in testset])
    predictions = []
    for row, vector in zip(testset, utterance_vectors):
        predicted, score, margin = predict_route(
            vector, exemplar_matrix, route_slices, aggregation
        )
        predictions.append(
            Prediction(
                row["utterance"], row["lang"], row["expected_route"], predicted, score, margin
            )
        )
    elapsed = time.perf_counter() - start

    result = summarize(predictions, elapsed)
    result["embedder"] = embedder_name
    result["pooling"] = pooling if embedder_name == "static" else "n/a"
    result["aggregation"] = aggregation
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
    parser.add_argument("--output")
    arguments = parser.parse_args()

    result = evaluate(
        arguments.embedder, arguments.model_dir, arguments.pooling, arguments.aggregation
    )

    print(
        f"embedder={result['embedder']} pooling={result['pooling']} "
        f"aggregation={result['aggregation']}"
    )
    print(f"  route accuracy   {result['route_accuracy']:.3f}")
    print(f"  tool accuracy    {result['tool_accuracy']:.3f}")
    print(f"  en / ja          {result['route_accuracy_en']:.3f} / {result['route_accuracy_ja']:.3f}")
    print(f"  mean score corr  {result['mean_top_score_correct']:.3f}")
    print(f"  mean score esc   {result['mean_top_score_escape']:.3f}")
    print(f"  escape 100% pt   {result['full_escape_operating_point']}")
    print(f"  sec/utterance    {result['seconds_per_utterance']:.5f}")

    if arguments.output:
        output_path = Path(arguments.output)
        if not output_path.is_absolute():
            output_path = SCRIPT_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
