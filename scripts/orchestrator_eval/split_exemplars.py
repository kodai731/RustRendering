"""Deterministic train/val split of exemplars.jsonl.

For each unique route, extract a fixed number of en and ja rows into the
validation set. The remainder is the training set. Both are written as JSONL
in the same {route, lang, utterance} format as the source file.

Run:
    python scripts/orchestrator_eval/split_exemplars.py
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from dataset import read_jsonl

SCRIPT_DIR = Path(__file__).resolve().parent


def split_exemplars(rows: list[dict], seed: int, per_lang: dict[str, int]) -> tuple[list[dict], list[dict]]:
    """Split exemplar rows into train and val sets.

    For each unique route, draw exactly per_lang[lang] rows of that language
    into the validation set. The remainder goes to training.

    Returns (train_rows, val_rows).
    """
    rng = random.Random(seed)

    # Group rows by (route, lang)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["route"], row["lang"])
        groups[key].append(row)

    train_rows: list[dict] = []
    val_rows: list[dict] = []

    for key, group_rows in groups.items():
        route, lang = key
        n_val = per_lang.get(lang, 0)
        chosen = rng.sample(group_rows, min(n_val, len(group_rows)))
        chosen_set = frozenset(id(r) for r in chosen)
        for row in group_rows:
            if id(row) in chosen_set:
                val_rows.append(row)
            else:
                train_rows.append(row)

    return train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Split exemplars.jsonl into train and val sets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    parser.add_argument(
        "--per-lang",
        type=json.loads,
        default='{"en": 2, "ja": 3}',
        help="JSON dict of language -> val count (default: {\"en\": 2, \"ja\": 3})",
    )
    args = parser.parse_args()

    source_path = SCRIPT_DIR / "exemplars.jsonl"
    train_path = SCRIPT_DIR / "exemplars_train.jsonl"
    val_path = SCRIPT_DIR / "val_routes.jsonl"

    rows = read_jsonl(source_path)
    train_rows, val_rows = split_exemplars(rows, args.seed, args.per_lang)

    # Write outputs
    train_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n",
        encoding="utf-8",
    )
    val_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val_rows) + "\n",
        encoding="utf-8",
    )

    # Build count table: per route, show total / train / val
    source_counts = defaultdict(lambda: defaultdict(int))
    train_counts = defaultdict(lambda: defaultdict(int))
    val_counts = defaultdict(lambda: defaultdict(int))

    for row in rows:
        source_counts[row["route"]][row["lang"]] += 1
    for row in train_rows:
        train_counts[row["route"]][row["lang"]] += 1
    for row in val_rows:
        val_counts[row["route"]][row["lang"]] += 1

    all_routes = sorted(source_counts.keys())
    print(f"{'route':<30} {'source':>7} {'train':>7} {'val':>7}")
    print("-" * 55)
    for route in all_routes:
        s_total = sum(source_counts[route].values())
        t_total = sum(train_counts[route].values())
        v_total = sum(val_counts[route].values())
        print(f"{route:<30} {s_total:>7} {t_total:>7} {v_total:>7}")

    total_source = len(rows)
    total_train = len(train_rows)
    total_val = len(val_rows)
    print("-" * 55)
    print(f"{'TOTAL':<30} {total_source:>7} {total_train:>7} {total_val:>7}")

    # Verification
    train_set = frozenset(json.dumps(r, ensure_ascii=False) for r in train_rows)
    val_set = frozenset(json.dumps(r, ensure_ascii=False) for r in val_rows)
    source_set = frozenset(json.dumps(r, ensure_ascii=False) for r in rows)

    intersection = train_set & val_set
    union = train_set | val_set

    print()
    if intersection:
        print(f"FAIL: train ∩ val is not empty ({len(intersection)} overlapping rows)")
        exit(1)
    else:
        print("OK: train ∩ val = empty")

    if union != source_set:
        missing = source_set - union
        extra = union - source_set
        print(f"FAIL: train ∪ val ≠ original (missing={len(missing)}, extra={len(extra)})")
        exit(1)
    else:
        print("OK: train ∪ val = original file contents")


if __name__ == "__main__":
    main()
