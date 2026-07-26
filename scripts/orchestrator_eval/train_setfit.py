"""Contrastively adapt the sentence encoder to the route set, then export it.

The held-out failures are dominated by enum siblings (start vs end, show vs hide)
that cosine similarity to route-name paraphrases cannot separate. SetFit trains
the encoder on same-route / different-route pairs, which makes every sibling
route an explicit negative, and fits a logistic head on the adapted embeddings.

Trains on exemplars.jsonl only -- the same index the router already carries, so
no new labelling. `heldout.jsonl` is never read here.

Run:
    .venv-setfit/bin/python scripts/orchestrator_eval/train_setfit.py \
        --output-dir dist/orchestrator_router/e5-small-route-setfit --epochs 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import read_jsonl  # noqa: E402
from local_paths import find_model_dir  # noqa: E402

BASE_MODEL_NAME = "multilingual-e5-small"
E5_QUERY_PREFIX = "query: "
SEED = 20260726


def build_training_dataset(exemplars: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [E5_QUERY_PREFIX + row["utterance"] for row in exemplars],
            "label": [row["route"] for row in exemplars],
        }
    )


def train_route_encoder(
    exemplars: list[dict],
    base_model_dir: str,
    epochs: int,
    batch_size: int,
    output_dir: Path,
    save_strategy: str = "no",
) -> SetFitModel:
    """`save_strategy="epoch"` is what `select_setfit_epochs.py` needs.

    Comparing epoch counts from checkpoints of one run costs a single training
    trajectory instead of one run per candidate, and the candidates are nested
    rather than independently seeded, so the comparison is between epochs and not
    between initializations. That only works with `save_total_limit` lifted —
    SetFit defaults it to 1, which leaves the last epoch and deletes the rest.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = SetFitModel.from_pretrained(base_model_dir, use_differentiable_head=False)
    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        batch_size=batch_size,
        num_epochs=epochs,
        num_iterations=20,
        save_strategy=save_strategy,
        save_total_limit=None,
        seed=SEED,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=build_training_dataset(exemplars),
        column_mapping={"text": "text", "label": "label"},
    )
    trainer.train()
    return model


def export_encoder_to_onnx(encoder, output_dir: Path) -> None:
    """Write the encoder in the layout ContextualEmbedder already reads.

    Keeping the ONNX contract identical -- last_hidden_state, mean pooled and
    normalised on the numpy side -- is what lets the adapted model be measured
    by the unchanged evaluation driver rather than a parallel one.
    """
    transformer = encoder[0].auto_model
    tokenizer = encoder.tokenizer
    transformer.eval()

    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_dir))

    sample = tokenizer(["query: sample utterance"], return_tensors="pt", padding=True)
    torch.onnx.export(
        transformer,
        (sample["input_ids"], sample["attention_mask"]),
        str(onnx_dir / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        dynamo=False,
    )


def save_classification_head(model: SetFitModel, output_dir: Path) -> None:
    head = model.model_head
    payload = {
        "classes": list(head.classes_),
        "coefficients": head.coef_.tolist(),
        "intercepts": head.intercept_.tolist(),
    }
    (output_dir / "route_head.json").write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", default=find_model_dir(BASE_MODEL_NAME))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--export-only", action="store_true")
    arguments = parser.parse_args()

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if arguments.export_only:
        model = SetFitModel.from_pretrained(str(output_dir / "setfit"))
    else:
        exemplars = read_jsonl(SCRIPT_DIR / "exemplars.jsonl")
        routes = len(set(row["route"] for row in exemplars))
        print(f"training on {len(exemplars)} exemplars over {routes} routes")
        model = train_route_encoder(
            exemplars,
            arguments.base_model_dir,
            arguments.epochs,
            arguments.batch_size,
            output_dir,
        )
        model.save_pretrained(str(output_dir / "setfit"))
    export_encoder_to_onnx(model.model_body, output_dir)
    save_classification_head(model, output_dir)
    print(f"wrote adapted encoder and head to {output_dir}")


if __name__ == "__main__":
    main()
