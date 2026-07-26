# Route encoder variants for the orchestrator

Encoders adapted to the 29-route set for `src/orchestrator/`. Every variant here
was trained by `scripts/orchestrator_eval/` on `exemplars.jsonl` alone, and
measured on `heldout.jsonl`, which no training run has ever read.

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

Measured on held-out (128 cases), keyword router off, cosine to nearest exemplar.

| Directory | Training | route | tool | enum誤り | escape retained |
|---|---|---|---|---|---|
| — (base, not stored) | none | 0.603 | 0.698 | 11 | 0.286 |
| `setfit-1ep` | SetFit, 435 steps | 0.629 | 0.750 | 14 | 0.521 |
| `setfit-3ep` | SetFit, 1305 steps | 0.698 | 0.784 | 10 | 0.519 |
| `setfit-6ep` | SetFit, 2610 steps | **0.724** | **0.793** | **8** | **0.524** |
| `sibling-hardneg` | MNRL on sibling triplets, 430 steps | 0.698 | 0.767 | **8** | 0.444 |

`setfit-6ep` is the best measured, but **the epoch count is not yet validated**:
devset saturates at 0.877 for both 3ep and 6ep, so the ranking above comes from
held-out numbers. That makes 0.724 an optimistic estimate of unseen performance.
Carving a validation split out of `exemplars.jsonl` is the fix, and until that
runs, treat the epoch choice as provisional.

`sibling-hardneg` and the `route_head.json` classifier were both **rejected** —
kept here because the measurements that rejected them are cited in the design
docs. See `setfit_eval.md` under the design directory.

## Layout

```
setfit-<n>ep/
  onnx/model.onnx        engine input; last_hidden_state, pooled on the caller side
  setfit/                torch weights, for re-export or further training
  route_head.json        logistic head coefficients (measured, not adopted)
  tokenizer.json         copy of the base tokenizer
sibling-hardneg/
  sentence_transformer/  torch weights (no SetFit head in this variant)
```

## Reproduce

```bash
.venv-setfit/bin/python scripts/orchestrator_eval/train_setfit.py \
    --output-dir models/gemma/setfit-6ep --epochs 6

.venv-orchestrator-eval/bin/python scripts/orchestrator_eval/eval_router.py \
    --embedder contextual --keyword-router off --dataset heldout \
    --model-dir models/gemma/setfit-6ep
```

Training is CPU-only and deterministic under the seed in `train_setfit.py`:
about 3 minutes per epoch. The ONNX export matches the torch model to
cos = 1.0 (max abs diff 2.0e-7).
