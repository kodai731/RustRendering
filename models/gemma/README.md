# Route encoder variants for the orchestrator

Encoders adapted to the 29-route set for `src/orchestrator/`. The adopted variant here was trained by AnimationModelTraining's `scripts/orchestrator_router/` on `exemplars.jsonl` alone, and measured on `heldout.jsonl`, which no training run has ever read.

## What is versioned and what is not

Weights are too large to version, so this directory tracks the files that
explain how each model was made and leaves the weights untracked beside them.

| Tracked | Untracked (see `.gitignore`) |
|---|---|
| this README, configs, `route_head.json` | `onnx/model.onnx`, `model.safetensors`, `*.pkl`, `tokenizer.json` |

`tokenizer.json` is a verbatim copy of the base encoder's and 17MB apiece, so it
is excluded rather than versioned seven times. Re-run the training command below
to regenerate any untracked file; nothing here is hand-edited.

Base encoder: `intfloat/multilingual-e5-small`, resolved through
`ModelStoragePath` in `.claude/local/paths.md`. Inputs carry the `query: `
prefix, which E5 requires even for symmetric tasks.

## Variants

This repo only contains the adopted variant. Experimental variants have been moved to ModelStoragePath/orchestrator-router/ (the `models/` directory of the CodeAgent repository). The `e5-raw/` directory remains in this repo as it is a runtime artifact loaded by the AND gate. The full list of variants and retraining procedures are managed in AnimationModelTraining's `scripts/orchestrator_router/`.

`setfit-3ep-p2` is the variant to load (adopted 2026-07-28; retrained on 1093 exemplars incl. 397 synthesized paraphrases (undo/redo と 一つ 系の境界汚染 32 件を剪定済み), epoch 3 chosen on a validation split; AND-gate thresholds 0.93/0.90, delta 0.0025).

## Reproduce

```bash
# run from the AnimationModelTraining checkout
.venv-setfit-container/bin/python scripts/orchestrator_router/train_setfit.py \
    --output-dir <this repo>/models/gemma/setfit-3ep-p2 --epochs 3

.venv-setfit-container/bin/python scripts/orchestrator_router/eval_router.py \
    --model-dir <this repo>/models/gemma/setfit-3ep-p2
```
```

Training is CPU-only and deterministic under the seed in `train_setfit.py`:
about 4 minutes per epoch at 464 exemplars. The ONNX export matches the torch
model to cos = 1.0 (max abs diff 2.0e-7).
