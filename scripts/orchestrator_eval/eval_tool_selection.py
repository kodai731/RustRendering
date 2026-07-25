"""Measure how accurately Gemma 3 270M picks a tool under grammar constraint.

Run before committing to the Rust decode loop: if tool selection does not hold
up here, the Rust work is wasted. Usage:

    .venv-orchestrator-eval/bin/python scripts/orchestrator_eval/eval_tool_selection.py \
        --variant baseline --output results/baseline.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from gemma_session import GemmaOnnxSession
from prompt_builder import DEMONSTRATED_TOOLS, PROMPT_STRATEGIES
from tool_schema import TOOL_VARIANTS, build_json_schema, find_tool

DEFAULT_MODEL_DIR = "/home/kodai/Projects/CodeAgent/models/gemma-3-270m-it-ONNX"

PLAYBACK_ACTION_BY_TOOL = {
    "play_animation": "play",
    "pause_animation": "pause",
    "stop_animation": "stop",
}


def load_testset(path: Path) -> list[dict]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def adapt_case_to_variant(case: dict, variant: str) -> dict:
    action = PLAYBACK_ACTION_BY_TOOL.get(case["tool"])
    if variant != "aggregated_playback" or action is None:
        return case
    return {**case, "tool": "control_playback", "args": {"action": action}}


def score_arguments(expected: dict, predicted: dict) -> tuple[int, int]:
    scored = {name: value for name, value in expected.items() if value is not None}
    if not scored:
        return 0, 0
    matched = sum(1 for name, value in scored.items() if predicted.get(name) == value)
    return matched, len(scored)


def evaluate_case(session: GemmaOnnxSession, build_prompt, tools, grammar: str, case: dict) -> dict:
    rendered = build_prompt(tools, case["utterance"])
    prompt = session.render_prompt(rendered.system, rendered.user)
    result = session.generate_constrained(prompt, grammar)

    try:
        predicted = json.loads(result.text)
    except json.JSONDecodeError:
        predicted = {}

    predicted_tool = predicted.get("tool", "")
    matched_args, scored_args = score_arguments(case["args"], predicted)

    return {
        "lang": case["lang"],
        "utterance": case["utterance"],
        "expected_tool": case["tool"],
        "predicted_tool": predicted_tool,
        "tool_correct": predicted_tool == case["tool"],
        "demonstrated": case["tool"] in DEMONSTRATED_TOOLS,
        "predicted_args": {k: v for k, v in predicted.items() if k != "tool"},
        "matched_args": matched_args,
        "scored_args": scored_args,
        "raw": result.text,
        "stop_reason": result.stop_reason,
        "token_count": result.token_count,
        "prefill_ms": result.prefill_seconds * 1000.0,
        "ms_per_token": result.milliseconds_per_token(),
        "ms_per_mask": result.milliseconds_per_mask(),
    }


def summarize(records: list[dict], tools) -> dict:
    total = len(records)
    correct = sum(1 for record in records if record["tool_correct"])

    args_matched = sum(record["matched_args"] for record in records if record["tool_correct"])
    args_scored = sum(record["scored_args"] for record in records if record["tool_correct"])

    per_language = {}
    for language in sorted({record["lang"] for record in records}):
        subset = [record for record in records if record["lang"] == language]
        hits = sum(1 for record in subset if record["tool_correct"])
        per_language[language] = {"total": len(subset), "correct": hits, "accuracy": hits / len(subset)}

    per_kind = {}
    for kind in ("query", "command", "escape"):
        subset = [
            record for record in records
            if (found := find_tool(tools, record["expected_tool"])) is not None and found.kind == kind
        ]
        if subset:
            hits = sum(1 for record in subset if record["tool_correct"])
            per_kind[kind] = {"total": len(subset), "correct": hits, "accuracy": hits / len(subset)}

    held_out = [record for record in records if not record["demonstrated"]]
    held_out_correct = sum(1 for record in held_out if record["tool_correct"])

    invalid_json = sum(1 for record in records if not record["predicted_tool"])
    predicted_distribution = Counter(record["predicted_tool"] for record in records)

    return {
        "total": total,
        "tool_accuracy": correct / total,
        "held_out_total": len(held_out),
        "held_out_accuracy": (held_out_correct / len(held_out)) if held_out else None,
        "arg_accuracy": (args_matched / args_scored) if args_scored else None,
        "arg_scored_count": args_scored,
        "invalid_json": invalid_json,
        "per_language": per_language,
        "per_kind": per_kind,
        "predicted_distribution": predicted_distribution.most_common(),
        "mean_prefill_ms": sum(r["prefill_ms"] for r in records) / total,
        "mean_ms_per_token": sum(r["ms_per_token"] for r in records) / total,
        "mean_ms_per_mask": sum(r["ms_per_mask"] for r in records) / total,
    }


def print_summary(variant: str, strategy: str, summary: dict, records: list[dict]) -> None:
    print(f"\n=== variant: {variant} / prompt: {strategy} ===")
    print(f"tool accuracy : {summary['tool_accuracy']:.1%} ({summary['total']} cases)")
    if summary["held_out_accuracy"] is not None:
        print(f"  excl. demoed: {summary['held_out_accuracy']:.1%} ({summary['held_out_total']} cases)")
    if summary["arg_accuracy"] is not None:
        print(f"arg  accuracy : {summary['arg_accuracy']:.1%} ({summary['arg_scored_count']} scored args)")
    print(f"invalid json  : {summary['invalid_json']}")

    for language, stats in summary["per_language"].items():
        print(f"  lang {language}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    for kind, stats in summary["per_kind"].items():
        print(f"  kind {kind}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    print(f"prefill {summary['mean_prefill_ms']:.0f}ms  "
          f"decode {summary['mean_ms_per_token']:.1f}ms/tok  "
          f"mask {summary['mean_ms_per_mask']:.2f}ms/tok")

    print("\nmost predicted tools:")
    for name, count in summary["predicted_distribution"][:8]:
        print(f"  {count:3d}  {name or '<invalid>'}")

    print("\nfailures:")
    for record in records:
        if not record["tool_correct"]:
            print(f"  [{record['lang']}] {record['utterance']!r}")
            print(f"        expected {record['expected_tool']} / got {record['raw']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="baseline", choices=sorted(TOOL_VARIANTS))
    parser.add_argument("--prompt", default="few_shot", choices=sorted(PROMPT_STRATEGIES))
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--testset", default=str(Path(__file__).parent / "testset.jsonl"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()

    tools = TOOL_VARIANTS[arguments.variant]
    build_prompt = PROMPT_STRATEGIES[arguments.prompt]
    session = GemmaOnnxSession(arguments.model_dir)
    grammar = session.build_grammar(build_json_schema(tools))

    cases = load_testset(Path(arguments.testset))
    if arguments.limit:
        cases = cases[: arguments.limit]

    records = []
    for index, case in enumerate(cases, start=1):
        adapted = adapt_case_to_variant(case, arguments.variant)
        record = evaluate_case(session, build_prompt, tools, grammar, adapted)
        records.append(record)
        mark = "ok " if record["tool_correct"] else "NG "
        print(f"{index:3d}/{len(cases)} {mark}{record['expected_tool']:22s} <- {record['raw']}", flush=True)

    summary = summarize(records, tools)
    print_summary(arguments.variant, arguments.prompt, summary, records)

    if arguments.output:
        output_path = Path(arguments.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "variant": arguments.variant,
            "prompt": arguments.prompt,
            "summary": summary,
            "records": records,
        }
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
