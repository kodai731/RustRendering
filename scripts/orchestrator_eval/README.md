# Orchestrator tool-selection evaluation

Measures how accurately the orchestrator picks an editor tool from a user
utterance, on plain onnxruntime.

| Driver | Approach | Result on `heldout` |
|---|---|---|
| `eval_tool_selection.py` | Generative — Gemma 3 270M under llguidance JSON Schema constraint | not yet re-measured |
| `eval_router.py` | Embedding router — cosine similarity against per-route exemplars, then the polarity tie-break | 84.5% route accuracy on the SetFit-adapted encoder, 80.2% with `--stage-a none` |

## Which Stage A

The keyword rule table that used to run in front of the router is gone. It decided
routes outright and so bypassed the rejection threshold, and on `heldout` it fired
for 47 of 128 utterances at 0.468 precision. Twelve of its rules were right every
time they fired and changed nothing; the damage came from `play_animation` (16 fires,
2 right), `stop_animation` (5, 0) and `toggle_loop` (4, 1), whose trigger words are
the domain's most common. Interrogative narrowing went with it: it detected 6 of the
16 query utterances and broke two correct edit routes, so it was neither a guarantee
nor a gain.

| Stage A | route | tool | wrong executed at 100% escape recall |
|---|---|---|---|
| `none` | 0.802 | 0.862 | 0.026 |
| `polarity` | **0.845** | **0.871** | **0.009** |
| retired rules | 0.655 | 0.698 | 0.241 |

`polarity` reorders the encoder's own top two when they are declared opposite poles
of one axis and the utterance names the runner-up's pole. Five held-out cases flip,
all of them correct, three of which the encoder had separated by 0.005 cosine or
less. Read `polarity.py` before editing `polarity_groups.json` — that file is
generated.

## Which dataset to quote

`devset.jsonl` is the corpus the retired rules were written against, so it reads
high for reasons that do not carry over: those rules scored 1.000 there. It is now
the selection set for any parameter that has to be chosen. Quote `heldout.jsonl`;
quote `devset` only alongside it, as the size of the fit.

| Dataset | Cases | Route accuracy (`none`) | Route accuracy (`polarity`) |
|---|---|---|---|
| `devset` | 74 | 0.938 | 0.969 |
| `heldout` | 128 | 0.802 | 0.845 |

This is a test-vector generator and a go/no-go gate for `src/orchestrator/` —
not product code. Findings live in `${RustRenderingDocPath}/Design/`:
`20260725_gemma_270m_tool_selection_eval/` (generative) and
`20260725_tiny_llm_command_orchestrator/.../embedding_router.md` (router).

## Setup

```bash
uv venv .venv-orchestrator-eval
VIRTUAL_ENV=.venv-orchestrator-eval uv pip install \
    onnxruntime numpy tokenizers huggingface_hub llguidance jinja2
```

The model must be under `${ModelStoragePath}` (see `.claude/local/paths.md`):

```bash
.venv-orchestrator-eval/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('onnx-community/gemma-3-270m-it-ONNX',
    local_dir='<ModelStoragePath>/gemma-3-270m-it-ONNX',
    allow_patterns=['config.json','tokenizer.json','tokenizer_config.json',
                    'generation_config.json','chat_template.jinja',
                    'onnx/model.onnx','onnx/model.onnx_data'])"
```

The router additionally needs `multilingual-e5-small` and the `onnx` package:

```bash
VIRTUAL_ENV=.venv-orchestrator-eval uv pip install onnx
.venv-orchestrator-eval/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('intfloat/multilingual-e5-small',
    local_dir='<ModelStoragePath>/multilingual-e5-small',
    allow_patterns=['config.json','tokenizer.json','tokenizer_config.json',
                    'special_tokens_map.json','onnx/*'])"
```

## Run

```bash
.venv-orchestrator-eval/bin/python scripts/orchestrator_eval/eval_router.py \
    --embedder contextual --model-dir models/gemma/setfit-6ep-en8 \
    --output results/stage_a_polarity_heldout.json

.venv-orchestrator-eval/bin/python scripts/orchestrator_eval/eval_tool_selection.py \
    --variant baseline --prompt few_shot --output results/few_shot.json
```

Regenerate the polarity table after any edit to `exemplars.jsonl`, then re-export
the cases the Rust tie-break is tested against. `cargo test --lib` fails naming
these two commands if the export is stale, so neither can be forgotten silently:

```bash
.venv-orchestrator-eval/bin/python scripts/orchestrator_eval/derive_polarity_terms.py
.venv-orchestrator-eval/bin/python scripts/orchestrator_eval/export_tiebreak_cases.py
```

Re-run the sweep as well if the derivation itself changed:

```bash
.venv-orchestrator-eval/bin/python scripts/orchestrator_eval/sweep_polarity.py \
    --model-dir models/gemma/setfit-6ep-en8 --output results/polarity_sweep.json
```

## Handing the router to the engine

`src/orchestrator/systems/router.rs` ranks the routes and
`thyllore_ml_core::sentence_encoder` encodes the utterance, but the accuracy above
belongs to this driver. What ties the two together is an export of the exemplar
vectors plus the decision this driver reached for every labelled utterance, which
the Rust side replays through its own encoder and ranker:

```bash
.venv-orchestrator-eval/bin/python scripts/orchestrator_eval/export_router_index.py \
    --model-dir models/gemma/setfit-6ep-en8

THYLLORE_ROUTER_MODEL_DIR=$PWD/models/gemma/setfit-6ep-en8 \
    cargo test --test orchestrator_router_parity
```

All 202 utterances agree on route, and on score within 1e-3 — the two runtimes are
different onnxruntime builds. Re-export after retraining or after any change to
`exemplars.jsonl`; the test skips with a message when the model directory is unset,
so a stale export shows up as a disagreement rather than as a silent pass.

The engine's rejection threshold is `DEFAULT_REJECTION_THRESHOLD`, selected on
`devset` as the lowest value that executes no wrong route there. It does not
transfer intact — see the constant's own documentation for what `heldout` says
about it.

| Flag | Driver | Values |
|---|---|---|
| `--dataset` | both | `heldout` (default), `devset` |
| `--embedder` | router | `static` (Gemma embedding table), `contextual` (e5-small) |
| `--pooling` | router | `mean`, `unit_mean`, `sif` — `static` only |
| `--aggregation` | router | `max`, `mean_top3` — how per-route exemplar scores combine |
| `--stage-a` | router | `polarity` (default), `none` — the deterministic stage after ranking |
| `--variant` | generative | `baseline`, `aggregated_playback` |
| `--prompt` | generative | `zero_shot`, `request_framing`, `few_shot` |
| `--model-dir` | both | defaults per embedder / the Gemma 3 270M ONNX directory |
| `--limit` | generative | evaluate the first N cases only |

A full router run is under a second. The generative driver costs about 2 s per
utterance on a single intra-op thread, so a 128-case run takes ~4 minutes.

## Files

| File | Role |
|---|---|
| `tool_schema.py` | Tool definitions. Single source for the JSON Schema, the prompt catalog and the routes |
| `route_schema.py` | Route expansion: tool × the enum arguments that appear in an utterance |
| `dataset.py` | Dataset name to path, and what each set is allowed to claim |
| `normalize.py` | Utterance normalization, mirroring `src/orchestrator/systems/normalize.rs` |
| `polarity.py` | Polarity axes, term derivation, and the tie-break the engine mirrors |
| `derive_polarity_terms.py` | Writes `src/orchestrator/data/polarity_groups.json` from the exemplars |
| `export_tiebreak_cases.py` | Records this tie-break's answers over every utterance, for the Rust differential test |
| `export_router_index.py` | Writes the engine's exemplar vectors and the decisions the Rust router is held to |
| `sweep_polarity.py` | Evidence for the minimum-support and margin parameters the tie-break does not have |
| `exemplars.jsonl` | 696 example utterances (29 routes × 8 English + 16 Japanese). The router's index; disjoint from both datasets |
| `train_setfit.py` | Contrastively adapts the encoder to the routes and exports it as ONNX |
| `select_setfit_epochs.py` | Picks the epoch count on a split of the exemplars, one checkpointed run per candidate |
| `static_embedder.py` | Sentence vectors pooled from Gemma's input embedding table alone |
| `contextual_embedder.py` | Sentence vectors from an ONNX sentence encoder |
| `prompt_builder.py` | Prompt strategies. Few-shot examples follow the tool variant |
| `gemma_session.py` | ONNX session, chat template, KV cache decode loop, llguidance mask |
| `eval_router.py` | Router evaluation driver, threshold sweep and metrics |
| `eval_tool_selection.py` | Generative evaluation driver and metrics |
| `devset.jsonl` | 74 utterances (56 English, 18 Japanese). The retired keyword rules were authored against these; now the selection set |
| `heldout.jsonl` | 128 utterances (64 English, 64 Japanese), 29 routes × 4 plus 12 escape. Written without reading any keyword table |

Arguments that the utterance does not determine are written as `null` and
excluded from argument scoring. The escape-hatch cases carry no route and must
fall below the router's rejection threshold instead.

Keeping `heldout.jsonl` honest means never consulting `polarity_groups.json` or
`exemplars.jsonl` while editing it, and never editing a case because the router got
it wrong — only because the label itself is wrong or the utterance has two valid
readings.
