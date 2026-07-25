# Orchestrator tool-selection evaluation

Measures how accurately the orchestrator picks an editor tool from a user
utterance, on plain onnxruntime. Two approaches share one testset:

| Driver | Approach | Result |
|---|---|---|
| `eval_tool_selection.py` | Generative — Gemma 3 270M under llguidance JSON Schema constraint | 43.2% tool accuracy, escape hatch 0/9 |
| `eval_router.py` | Embedding router — cosine similarity against per-route exemplars | 80.0% route accuracy, escape 9/9 at 92.3% retained |

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
| `--embedder` | router | `static` (Gemma embedding table), `contextual` (e5-small) |
| `--pooling` | router | `mean`, `unit_mean`, `sif` — `static` only |
| `--aggregation` | router | `max`, `mean_top3` — how per-route exemplar scores combine |
| `--variant` | generative | `baseline`, `aggregated_playback` |
| `--prompt` | generative | `zero_shot`, `request_framing`, `few_shot` |
| `--model-dir` | both | defaults per embedder / the Gemma 3 270M ONNX directory |
| `--limit` | generative | evaluate the first N cases only |

A full router run is under a second. The generative driver costs about 2 s per
utterance on a single intra-op thread, so its 74-case run takes ~3 minutes.

## Files

| File | Role |
|---|---|
| `tool_schema.py` | Tool definitions. Single source for the JSON Schema, the prompt catalog and the routes |
| `route_schema.py` | Route expansion: tool × the enum arguments that appear in an utterance |
| `exemplars.jsonl` | 232 example utterances (29 routes × 8, English and Japanese). Disjoint from the testset |
| `static_embedder.py` | Sentence vectors pooled from Gemma's input embedding table alone |
| `contextual_embedder.py` | Sentence vectors from an ONNX sentence encoder |
| `prompt_builder.py` | Prompt strategies. Few-shot examples follow the tool variant |
| `gemma_session.py` | ONNX session, chat template, KV cache decode loop, llguidance mask |
| `eval_router.py` | Router evaluation driver, threshold sweep and metrics |
| `eval_tool_selection.py` | Generative evaluation driver and metrics |
| `testset.jsonl` | 74 utterances (56 English, 18 Japanese) with expected tool and arguments |

Arguments that the utterance does not determine are written as `null` in the
testset and excluded from argument scoring. The 9 escape-hatch cases carry no
route and must fall below the router's rejection threshold instead.
