"""The labelled utterance sets both evaluation drivers read.

`devset` is what the keyword rules in `src/orchestrator/data/keyword_router_rules.json`
were authored against, so any rule-assisted number it produces is optimistic by
construction. `heldout` was written without reading that table and is the only
set whose accuracy is expected to hold on unseen phrasing. Report both, and
treat the gap between them as the amount of fitting the rules absorbed.
"""

import json
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent

DATASET_PATHS = {
    "devset": DATASET_DIR / "devset.jsonl",
    "heldout": DATASET_DIR / "heldout.jsonl",
}

DATASET_NAMES = tuple(DATASET_PATHS)
DEFAULT_DATASET = "heldout"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_dataset(name: str) -> list[dict]:
    return read_jsonl(DATASET_PATHS[name])
