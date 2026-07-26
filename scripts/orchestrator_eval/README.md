# Orchestrator tool-selection evaluation

Measures how accurately the orchestrator picks an editor tool from a user
utterance, on plain onnxruntime.

| Driver | Approach | Result on `heldout` |
|---|---|---|
| `eval_tool_selection.py` | Generative — Gemma 3 270M under llguidance JSON Schema constraint | not yet re-measured |
| `eval_router.py` | Embedding router — cosine similarity against per-route exemplars | 60.3% route accuracy, escape 12/12 at 28.6% retained |

## Which dataset to quote

`devset.jsonl` is the corpus the keyword rules were written against, so its
rule-assisted numbers are optimistic by construction — keyword precision reads
1.000 there and 0.468 on unseen phrasing. Quote `heldout.jsonl`; quote `devset`
only alongside it, as the size of the fit.

| Dataset | Cases | Keyword coverage | Keyword precision | Route accuracy (embed only) | Route accuracy (+keyword) |
|---|---|---|---|---|---|
| `devset` | 74 | 0.985 | 1.000 | 0.815 | 1.000 |
| `heldout` | 128 | 0.405 | 0.468 | 0.603 | 0.569 |

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
    --embedder contextual --aggregation max --output results/router_e5small_max.json

.venv-orchestrator-eval/bin/python scripts/orchestrator_eval/eval_tool_selection.py \
    --variant baseline --prompt few_shot --output results/few_shot.json
```

| Flag | Driver | Values |
|---|---|---|
| `--dataset` | both | `heldout` (default), `devset` |
| `--embedder` | router | `static` (Gemma embedding table), `contextual` (e5-small) |
| `--pooling` | router | `mean`, `unit_mean`, `sif` — `static` only |
| `--aggregation` | router | `max`, `mean_top3` — how per-route exemplar scores combine |
| `--keyword-router` | router | `on`, `off` — run the deterministic keyword rules first |
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
| `keyword_router.py` | Deterministic keyword routing. Reads `src/orchestrator/data/keyword_router_rules.json`, the same table the engine embeds |
| `exemplars.jsonl` | 348 example utterances (29 routes × 4 English + 8 Japanese). The router's index; disjoint from both datasets |
| `static_embedder.py` | Sentence vectors pooled from Gemma's input embedding table alone |
| `contextual_embedder.py` | Sentence vectors from an ONNX sentence encoder |
| `prompt_builder.py` | Prompt strategies. Few-shot examples follow the tool variant |
| `gemma_session.py` | ONNX session, chat template, KV cache decode loop, llguidance mask |
| `eval_router.py` | Router evaluation driver, threshold sweep and metrics |
| `eval_tool_selection.py` | Generative evaluation driver and metrics |
| `devset.jsonl` | 74 utterances (56 English, 18 Japanese). The keyword rules were authored against these |
| `heldout.jsonl` | 128 utterances (64 English, 64 Japanese), 29 routes × 4 plus 12 escape. Written without reading the keyword rule table |

Arguments that the utterance does not determine are written as `null` and
excluded from argument scoring. The escape-hatch cases carry no route and must
fall below the router's rejection threshold instead.

Keeping `heldout.jsonl` honest means never consulting
`keyword_router_rules.json` while editing it, and never editing a case because
the router got it wrong — only because the label itself is wrong or the
utterance has two valid readings.
