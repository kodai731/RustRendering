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
| — (base, not stored) | none | 0.690 | 0.784 | — | 0.363 |
| `setfit-6ep-en8` | SetFit, 3480 steps, 464 exemplars | **0.802** | **0.862** | **7** | 0.366 |
| `setfit-1ep` | SetFit, 435 steps, 348 exemplars | 0.629 | 0.750 | 14 | 0.521 |
| `setfit-3ep` | SetFit, 1305 steps, 348 exemplars | 0.698 | 0.784 | 10 | 0.519 |
| `setfit-6ep` | SetFit, 2610 steps, 348 exemplars | 0.724 | 0.793 | 8 | 0.524 |
| `sibling-hardneg` | MNRL on sibling triplets, 430 steps | 0.698 | 0.767 | 8 | 0.444 |

The first two rows use the current 464-exemplar index (8 per language per route);
the rest were measured when English had only 4 per route, so compare them among
themselves. `setfit-6ep-en8` is the variant to load.

Its lower `escape retained` is not a regression: 9 of the 12 escape cases drop
below 0.73, and the operating point is pinned by three genuinely ambiguous ones
(`fix it`). At 10/12 escape recall it retains 0.710 against `setfit-6ep`'s 0.560.

**The epoch count is not yet validated**: devset saturates for both 3ep and 6ep,
so the ranking comes from held-out numbers, which makes 0.802 an optimistic
estimate of unseen performance. Carving a validation split out of
`exemplars.jsonl` is the fix; until that runs, treat the epoch choice as provisional.

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
    --output-dir models/gemma/setfit-6ep-en8 --epochs 6

.venv-orchestrator-eval/bin/python scripts/orchestrator_eval/eval_router.py \
    --embedder contextual --keyword-router off --dataset heldout \
    --model-dir models/gemma/setfit-6ep-en8
```

Training is CPU-only and deterministic under the seed in `train_setfit.py`:
about 4 minutes per epoch at 464 exemplars. The ONNX export matches the torch
model to cos = 1.0 (max abs diff 2.0e-7).
