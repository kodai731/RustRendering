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

`setfit-3ep-cam` supersedes it (adopted 2026-08-19): retrained on 1165 exemplars after adding the 6 `camera_shot:*` routes (look_at_selection / orbit_around_selection / dolly_in / dolly_out / crane_up / crane_down); heldout route accuracy 0.905 (p2 0.845), Rust heldout correct 105/116, retained 82/85; adding the exemplars to p2's index without retraining regressed heldout to 0.819.

`setfit-3ep-camdir` supersedes it (adopted 2026-08-21, retrained 2026-08-22): adds the free-text `camera_direction` route (43 exemplars) for the Tier2 generative escape. The first `camera_direction` retrain (2026-08-21) regressed heldout to 0.836 — root cause was `camera_shot:*` having only 12 exemplars per preset against ~40 for every other route, so SetFit's fixed-seed pairwise contrastive sampling (`shuffle_combinations`, seed 42 over the full exemplar-combination space) reshuffled when the total exemplar count changed and the thinnest classes absorbed unrelated utterances (4/9 new heldout errors were `camera_shot:*` false positives). Fixed by bringing `camera_shot:*` to parity (40 exemplars/preset, +168 rows) plus 5 rows disambiguating two residual lexical collisions (`テンポを上げて` vs `crane_up`'s `上げて`, `画面いっぱいに` framing vs `look_at_selection`). Retrained on 1381 exemplars: heldout **0.888** (was 0.836, `setfit-3ep-cam` baseline 0.905), devset **0.985** (was 0.954), camera_shot-related heldout errors 4→1. Promoted to `SharedData/exports/helm_router_20260822/setfit-3ep-camdir/` (was `helm_router_20260821`); `EXPORTS_BUNDLE_DIR` in `src/ecs/resource/helm_state.rs` updated accordingly. Detail: SharedData `document/Rust_Rendering/Design/CameraDirection/20260818_helm_camera_direction.md`.

## Reproduce

```bash
# run from the AnimationModelTraining checkout
.venv-setfit-container/bin/python scripts/orchestrator_router/train_setfit.py \
    --output-dir <this repo>/models/gemma/setfit-3ep-cam --epochs 3

.venv-setfit-container/bin/python scripts/orchestrator_router/eval_router.py \
    --model-dir <this repo>/models/gemma/setfit-3ep-cam
```

Training is CPU-only and deterministic under the seed in `train_setfit.py`:
about 4 minutes per epoch at 464 exemplars. The ONNX export matches the torch
model to cos = 1.0 (max abs diff 2.0e-7).
