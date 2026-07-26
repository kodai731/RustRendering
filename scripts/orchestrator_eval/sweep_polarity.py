"""Evidence for the two parameters the polarity tie-break does not have.

Both were real candidates: a minimum exemplar support for a derived term, and a
similarity margin below which the tie-break is allowed to act. This sweep is why
neither shipped. Support above 1 only ever deletes terms that were pulling their
weight, and the margin never prevented a wrong swap once groups were restricted to
genuine polarity axes — a threshold that changes no decision is worse than no
threshold, because it still has to be defended.

Selection runs on `devset`; the held-out column is printed beside it only so the
gap is visible. Encoding dominates the runtime and does not depend on either
parameter, so the utterances and exemplars are embedded once and reused.

Run:
    .venv-orchestrator-eval/bin/python scripts/orchestrator_eval/sweep_polarity.py \
        --model-dir models/gemma/setfit-6ep-en8
"""

import argparse
import json
from pathlib import Path

from contextual_embedder import ContextualEmbedder
from dataset import read_jsonl
from eval_router import (
    build_exemplar_index,
    load_labelled_utterances,
    rank_routes,
    score_margin,
)
from local_paths import find_model_dir
from normalize import normalize_utterance
from polarity import PolarityTieBreaker, TieBreak, derive_table
from route_schema import ESCAPE_ROUTE, build_routes

CONTEXTUAL_MODEL_NAME = "multilingual-e5-small"
SCRIPT_DIR = Path(__file__).resolve().parent

SUPPORT_VALUES = (1, 2, 3)
MARGIN_VALUES = (0.0, 0.01, 0.02, 0.04, 0.06, 0.10, 0.15, 0.25, 1.01)
DATASETS = ("devset", "heldout")


def rank_cases(embedder: ContextualEmbedder, dataset: str, exemplars: list[dict]) -> list[dict]:
    routes = build_routes()
    exemplar_matrix, route_slices = build_exemplar_index(
        embedder, exemplars, [route.route_id for route in routes]
    )

    cases = load_labelled_utterances(dataset)
    vectors = embedder.encode_batch([row["utterance"] for row in cases])
    return [
        {
            "normalized": normalize_utterance(row["utterance"]),
            "expected": row["expected_route"],
            "ranked": rank_routes(vector, exemplar_matrix, route_slices, "max"),
        }
        for row, vector in zip(cases, vectors)
    ]


class MarginGatedTieBreaker:
    """The rejected variant: only act when the encoder's own top two are close.

    Lives here rather than in `polarity.py` because it is the thing being argued
    against. A margin of 1.01 exceeds any reachable cosine gap and so reproduces
    the shipped, ungated behaviour.
    """

    def __init__(self, table: dict, margin: float):
        self.inner = PolarityTieBreaker(table)
        self.margin = margin

    def resolve(self, normalized: str, ranked: list[tuple[str, float]]) -> TieBreak:
        if ranked[0][1] - ranked[1][1] >= self.margin:
            return TieBreak(ranked[0][0], swapped=False)
        return self.inner.resolve(normalized, ranked)


def score_configuration(ranked_cases: list[dict], tie_breaker: MarginGatedTieBreaker) -> dict:
    routed = [case for case in ranked_cases if case["expected"] != ESCAPE_ROUTE]
    correct = 0
    swaps = 0
    good_swaps = 0

    for case in routed:
        outcome = tie_breaker.resolve(case["normalized"], case["ranked"])
        was_correct = case["ranked"][0][0] == case["expected"]
        is_correct = outcome.winner == case["expected"]
        correct += is_correct
        swaps += outcome.swapped
        good_swaps += outcome.swapped and is_correct and not was_correct

    return {
        "route_accuracy": correct / len(routed),
        "swaps": swaps,
        "good_swaps": good_swaps,
        "bad_swaps": swaps - good_swaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    model_dir = arguments.model_dir or find_model_dir(CONTEXTUAL_MODEL_NAME)
    embedder = ContextualEmbedder(model_dir)
    exemplars = read_jsonl(SCRIPT_DIR / "exemplars.jsonl")
    ranked_by_dataset = {
        dataset: rank_cases(embedder, dataset, exemplars) for dataset in DATASETS
    }

    rows = []
    for support in SUPPORT_VALUES:
        table = derive_table(exemplars, support)
        for margin in MARGIN_VALUES:
            tie_breaker = MarginGatedTieBreaker(table, margin)
            rows.append(
                {
                    "min_support": support,
                    "margin": margin,
                    **{
                        dataset: score_configuration(cases, tie_breaker)
                        for dataset, cases in ranked_by_dataset.items()
                    },
                }
            )

    selected = min(
        rows, key=lambda row: (-row["devset"]["route_accuracy"], row["margin"], row["min_support"])
    )

    print(f"model={Path(model_dir).name}  selection=devset")
    print(
        f"{'support':>7} {'margin':>6} | {'devset':>6} {'swap':>4} {'good':>4} {'bad':>4}"
        f" | {'heldout':>7} {'swap':>4} {'good':>4} {'bad':>4}"
    )
    for row in rows:
        marker = " <-" if row is selected else ""
        print(
            f"{row['min_support']:>7} {row['margin']:>6.2f} |"
            f" {row['devset']['route_accuracy']:>6.3f} {row['devset']['swaps']:>4}"
            f" {row['devset']['good_swaps']:>4} {row['devset']['bad_swaps']:>4} |"
            f" {row['heldout']['route_accuracy']:>7.3f} {row['heldout']['swaps']:>4}"
            f" {row['heldout']['good_swaps']:>4} {row['heldout']['bad_swaps']:>4}{marker}"
        )

    print(
        f"\nselected min_support={selected['min_support']} margin={selected['margin']}"
        f" -> devset {selected['devset']['route_accuracy']:.3f}"
        f" heldout {selected['heldout']['route_accuracy']:.3f}"
    )

    if arguments.output:
        output_path = Path(arguments.output)
        if not output_path.is_absolute():
            output_path = SCRIPT_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"selected": selected, "rows": rows}, ensure_ascii=False, indent=2)
        )


if __name__ == "__main__":
    main()
