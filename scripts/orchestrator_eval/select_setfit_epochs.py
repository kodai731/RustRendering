"""Choose the SetFit epoch count on a split of the exemplars, not on `heldout`.

epochs=6 was picked by watching `heldout` move, which makes 0.845 a number the
epoch count was fitted to rather than a prediction about unseen phrasing. This
selects on exemplars the encoder never indexed: four per route per language are
withheld, the rest become the index, and the epoch with the best route accuracy on
the withheld four wins. `heldout` is not read here.

One training run, checkpointed every epoch, rather than one run per candidate:
the candidates are then points on a single trajectory, so the comparison is
between epoch counts and not between initializations, and it costs an eighth of
the time.

The polarity table is re-derived from the training split for each evaluation. The
shipped table is derived from all 464 exemplars, so using it here would let a term
that exists only because a withheld utterance used it decide that same utterance.

Run:
    .venv-setfit/bin/python scripts/orchestrator_eval/select_setfit_epochs.py \
        --output-dir models/gemma/setfit-epoch-selection --max-epochs 8
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import read_jsonl  # noqa: E402
from eval_router import build_exemplar_index, rank_routes, score_margin  # noqa: E402
from local_paths import find_model_dir  # noqa: E402
from normalize import normalize_utterance  # noqa: E402
from polarity import PolarityTieBreaker, derive_table  # noqa: E402
from route_schema import build_routes  # noqa: E402
from train_setfit import BASE_MODEL_NAME, E5_QUERY_PREFIX, SEED, train_route_encoder  # noqa: E402

DEFAULT_HOLDOUT_PER_LANGUAGE = 2
DEFAULT_MAX_EPOCHS = 8
MIN_SUPPORT = 1
CHECKPOINT_PREFIX = "checkpoint-"


class TorchEncoder:
    """The evaluation driver's embedder surface, backed by torch instead of ONNX.

    Selection happens before any ONNX export exists. e5-small's sentence-transformers
    configuration is mean pooling followed by normalization, which is what
    `contextual_embedder.py` does by hand, so the two agree on what a vector is.
    """

    def __init__(self, model_dir: Path):
        self.model = SentenceTransformer(str(model_dir))

    def fit_corpus(self, texts: list[str]) -> None:
        return

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            [E5_QUERY_PREFIX + text for text in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def split_exemplars(
    exemplars: list[dict], per_language: int
) -> tuple[list[dict], list[dict]]:
    """Withholds the same count from every (route, language) group.

    An unstratified draw would leave some route short of exemplars and inflate its
    confusions for a reason that has nothing to do with the epoch count.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in exemplars:
        groups.setdefault((row["route"], row["lang"]), []).append(row)

    rng = random.Random(SEED)
    train: list[dict] = []
    validation: list[dict] = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda row: row["utterance"])
        withheld = set(rng.sample(range(len(members)), per_language))
        for index, row in enumerate(members):
            (validation if index in withheld else train).append(row)
    return train, validation


def measure_route_accuracy(
    encoder: TorchEncoder,
    train: list[dict],
    validation: list[dict],
    route_ids: list[str],
    breaker: PolarityTieBreaker,
) -> dict:
    matrix, slices = build_exemplar_index(encoder, train, route_ids)
    vectors = encoder.encode_batch([row["utterance"] for row in validation])

    correct = 0
    swapped = 0
    margins = []
    for row, vector in zip(validation, vectors):
        ranked = rank_routes(vector, matrix, slices, "max")
        outcome = breaker.resolve(normalize_utterance(row["utterance"]), ranked)
        correct += outcome.winner == row["route"]
        swapped += outcome.swapped
        margins.append(score_margin(ranked))

    return {
        "route_accuracy": correct / len(validation),
        "swapped": swapped,
        "mean_margin": float(np.mean(margins)),
    }


def find_epoch_checkpoints(output_dir: Path) -> list[Path]:
    """Checkpoint directories in training order, one per epoch.

    They are named by optimizer step, so the epoch each belongs to is its position
    in the step-sorted list — `save_strategy="epoch"` writes exactly one per epoch.
    """
    checkpoints = [
        path
        for path in (output_dir / "checkpoints").iterdir()
        if path.is_dir() and path.name.startswith(CHECKPOINT_PREFIX)
    ]
    return sorted(checkpoints, key=lambda path: int(path.name[len(CHECKPOINT_PREFIX) :]))


def pick_best_epoch(measurements: list[dict]) -> dict:
    """Best validation accuracy, and the fewest epochs among equals.

    A tie broken toward more training would be choosing on nothing measured. The
    un-adapted encoder is epoch 0 and is excluded: it is the reference that says
    whether the adaptation did anything, not a candidate to ship.
    """
    candidates = [row for row in measurements if row["epochs"] > 0]
    return max(candidates, key=lambda row: (row["route_accuracy"], -row["epochs"]))


def report(measurements: list[dict], best: dict) -> None:
    print(f"{'epochs':>7} {'val route':>10} {'swapped':>8} {'mean margin':>12}")
    for row in measurements:
        marker = " <-" if row["epochs"] == best["epochs"] else ""
        print(
            f"{row['epochs']:>7} {row['route_accuracy']:>10.3f}"
            f" {row['swapped']:>8} {row['mean_margin']:>12.4f}{marker}"
        )
    print(f"\nselected epochs={best['epochs']} at {best['route_accuracy']:.3f}")
    print("retrain on all exemplars with train_setfit.py --epochs " f"{best['epochs']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", default=find_model_dir(BASE_MODEL_NAME))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--holdout-per-language", type=int, default=DEFAULT_HOLDOUT_PER_LANGUAGE
    )
    parser.add_argument("--output")
    arguments = parser.parse_args()

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exemplars = read_jsonl(SCRIPT_DIR / "exemplars.jsonl")
    train, validation = split_exemplars(exemplars, arguments.holdout_per_language)
    route_ids = [route.route_id for route in build_routes()]
    breaker = PolarityTieBreaker(derive_table(train, MIN_SUPPORT))
    print(f"{len(train)} indexed, {len(validation)} withheld, {len(route_ids)} routes")

    train_route_encoder(
        train,
        arguments.base_model_dir,
        arguments.max_epochs,
        arguments.batch_size,
        output_dir,
        save_strategy="epoch",
    )

    measurements = []
    for epoch, checkpoint in enumerate(
        [Path(arguments.base_model_dir), *find_epoch_checkpoints(output_dir)]
    ):
        encoder = TorchEncoder(checkpoint)
        measurement = measure_route_accuracy(encoder, train, validation, route_ids, breaker)
        measurements.append({"epochs": epoch, "checkpoint": checkpoint.name, **measurement})
        print(f"  epoch {epoch}: {measurement['route_accuracy']:.3f}")

    best = pick_best_epoch(measurements)
    report(measurements, best)

    if arguments.output:
        Path(arguments.output).write_text(
            json.dumps(
                {
                    "indexed": len(train),
                    "withheld": len(validation),
                    "holdout_per_language": arguments.holdout_per_language,
                    "selected_epochs": best["epochs"],
                    "measurements": measurements,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
