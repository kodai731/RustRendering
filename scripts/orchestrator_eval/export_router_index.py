"""Export the engine's route index, and what the Python router decides with it.

The engine cannot encode 464 exemplars at startup — that is seconds the editor
does not have — and re-encoding them would also let the index drift away from the
encoder the accuracy was measured with. Both problems go away if the vectors are
exported alongside the model that produced them.

Three files land in the model directory:

    router_index.json   route order and exemplar counts
    router_index.f32    the vectors, little-endian f32, rows in that order
    router_parity.json  the decision this driver reaches for every labelled
                        utterance, which tests/orchestrator_router_parity.rs
                        replays through the Rust encoder and ranker

The parity file is what makes the Rust port checkable. Rust reaching the same
route for all 202 utterances means its tokenizer, its pooling, its aggregation and
its tie-break all agree; a single differing step shows up as a differing route or
a score outside tolerance.

Run:
    .venv-orchestrator-eval/bin/python scripts/orchestrator_eval/export_router_index.py \
        --model-dir models/gemma/setfit-6ep-en8
"""

import argparse
import json
from pathlib import Path

import numpy as np

from contextual_embedder import ContextualEmbedder
from dataset import DATASET_NAMES, read_jsonl
from eval_router import (
    STAGE_A_MODES,
    StageA,
    build_exemplar_index,
    load_labelled_utterances,
    rank_routes,
    score_margin,
)
from route_schema import build_routes

SCRIPT_DIR = Path(__file__).resolve().parent

MANIFEST_FILENAME = "router_index.json"
VECTORS_FILENAME = "router_index.f32"
PARITY_FILENAME = "router_parity.json"


def build_manifest(route_slices: dict[str, slice], dimensions: int) -> dict:
    """Route order is the route table's, because ranking ties fall back to it.

    Python's sort and Rust's are both stable, so two routes on the same score come
    out in index order. That only matches across the two implementations while the
    index is written in the order both sides declare their routes, which is the
    order `build_exemplar_index` already put the matrix in.
    """
    return {
        "dimensions": dimensions,
        "routes": [
            {"route": route_id, "exemplars": span.stop - span.start}
            for route_id, span in route_slices.items()
        ],
    }


def verify_slices_tile_the_matrix(route_slices: dict[str, slice], rows: int) -> None:
    """The manifest describes the vectors by counts alone, so the rows have to be
    one contiguous run per route with no gap and nothing left over."""
    expected = 0
    for route_id, span in route_slices.items():
        if span.start != expected:
            raise ValueError(f"{route_id} starts at {span.start}, expected {expected}")
        expected = span.stop
    if expected != rows:
        raise ValueError(f"slices cover {expected} rows of {rows}")


def build_parity_cases(
    embedder: ContextualEmbedder,
    exemplar_matrix: np.ndarray,
    route_slices: dict[str, slice],
    stage_a: StageA,
) -> list[dict]:
    cases = []
    for dataset in DATASET_NAMES:
        rows = load_labelled_utterances(dataset)
        vectors = embedder.encode_batch([row["utterance"] for row in rows])

        for row, vector in zip(rows, vectors):
            ranked = rank_routes(vector, exemplar_matrix, route_slices, "max")
            predicted, score, swapped = stage_a.break_tie(row["utterance"], ranked)
            cases.append(
                {
                    "dataset": dataset,
                    "utterance": row["utterance"],
                    "route": predicted,
                    "score": score,
                    "margin": score_margin(ranked),
                    "swapped": swapped,
                }
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--stage-a", default="polarity", choices=STAGE_A_MODES)
    arguments = parser.parse_args()

    model_dir = Path(arguments.model_dir)
    route_ids = [route.route_id for route in build_routes()]
    exemplars = read_jsonl(SCRIPT_DIR / "exemplars.jsonl")

    embedder = ContextualEmbedder(model_dir)
    exemplar_matrix, route_slices = build_exemplar_index(embedder, exemplars, route_ids)
    ordered = np.ascontiguousarray(exemplar_matrix, dtype=np.float32)
    verify_slices_tile_the_matrix(route_slices, ordered.shape[0])

    manifest = build_manifest(route_slices, ordered.shape[1])
    (model_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (model_dir / VECTORS_FILENAME).write_bytes(ordered.tobytes(order="C"))

    cases = build_parity_cases(embedder, exemplar_matrix, route_slices, StageA(arguments.stage_a))
    (model_dir / PARITY_FILENAME).write_text(
        json.dumps(
            {
                "stage_a": arguments.stage_a,
                "exemplars": ordered.shape[0],
                "cases": cases,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"{model_dir} — {ordered.shape[0]} exemplars × {ordered.shape[1]} dimensions")
    print(f"  {len(manifest['routes'])} routes, {ordered.nbytes} vector bytes")
    print(f"  {len(cases)} parity cases, {sum(case['swapped'] for case in cases)} swapped")


if __name__ == "__main__":
    main()
