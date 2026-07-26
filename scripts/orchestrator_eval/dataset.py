"""The labelled utterance sets both evaluation drivers read.

`devset` is what the retired keyword rule table was authored against, so it reads
high for reasons that do not carry over — the rules scored 1.000 on it and 0.655 on
`heldout`. It is kept as the selection set for anything with a parameter to choose,
because using `heldout` for that would spend the only set whose accuracy is expected
to hold on unseen phrasing. Quote `heldout`; quote `devset` only beside it, as the
size of the fit.
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
