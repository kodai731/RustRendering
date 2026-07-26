"""Adapt the encoder with enum siblings as the explicit negative.

SetFit draws its negatives uniformly across routes, so for 29 routes a sibling
(`seek_time:start` against `seek_time:end`) lands in the negative slot about one
time in 28. Measured on held-out, that leaves the enum confusions untouched --
they are the failure the adaptation was supposed to remove. Here every triplet
pairs an anchor with a sibling of its own tool, so the gradient always points at
the distinction that cosine similarity cannot see.

Trains on exemplars.jsonl only. `heldout.jsonl` is never read here.

Run:
    .venv-setfit/bin/python scripts/orchestrator_eval/train_sibling_contrastive.py \
        --output-dir dist/orchestrator_router/e5-small-route-sibling --epochs 10
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import read_jsonl  # noqa: E402
from local_paths import find_model_dir  # noqa: E402
from train_setfit import (  # noqa: E402
    BASE_MODEL_NAME,
    E5_QUERY_PREFIX,
    SEED,
    export_encoder_to_onnx,
)

TRIPLETS_PER_ANCHOR = 4


def group_exemplars_by_route(exemplars: list[dict]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in exemplars:
        grouped[row["route"]].append(E5_QUERY_PREFIX + row["utterance"])
    return dict(grouped)


def find_sibling_routes(route_id: str, all_routes: list[str]) -> list[str]:
    tool_name = route_id.split(":")[0]
    return [
        other
        for other in all_routes
        if other != route_id and other.split(":")[0] == tool_name
    ]


def build_triplets(grouped: dict[str, list[str]], rng: random.Random) -> list[InputExample]:
    all_routes = list(grouped)
    triplets: list[InputExample] = []

    for route_id, utterances in grouped.items():
        siblings = find_sibling_routes(route_id, all_routes)
        negative_pool = [
            text
            for sibling in siblings
            for text in grouped[sibling]
        ] or [
            text
            for other in all_routes
            if other != route_id
            for text in grouped[other]
        ]

        for anchor in utterances:
            others = [text for text in utterances if text != anchor]
            for _ in range(TRIPLETS_PER_ANCHOR):
                triplets.append(
                    InputExample(
                        texts=[anchor, rng.choice(others), rng.choice(negative_pool)]
                    )
                )

    rng.shuffle(triplets)
    return triplets


def train_encoder(triplets: list[InputExample], base_model_dir: str, epochs: int, batch_size: int):
    torch.manual_seed(SEED)
    model = SentenceTransformer(base_model_dir)
    loader = DataLoader(triplets, shuffle=True, batch_size=batch_size, drop_last=True)
    model.fit(
        train_objectives=[(loader, losses.MultipleNegativesRankingLoss(model))],
        epochs=epochs,
        warmup_steps=int(0.1 * len(loader) * epochs),
        show_progress_bar=False,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", default=find_model_dir(BASE_MODEL_NAME))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    arguments = parser.parse_args()

    exemplars = read_jsonl(SCRIPT_DIR / "exemplars.jsonl")
    grouped = group_exemplars_by_route(exemplars)
    triplets = build_triplets(grouped, random.Random(SEED))
    print(f"{len(triplets)} triplets from {len(exemplars)} exemplars over {len(grouped)} routes")

    model = train_encoder(triplets, arguments.base_model_dir, arguments.epochs, arguments.batch_size)

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir / "sentence_transformer"))
    export_encoder_to_onnx(model, output_dir)
    print(f"wrote adapted encoder to {output_dir}")


if __name__ == "__main__":
    main()
