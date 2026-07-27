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
    "val": DATASET_DIR / "val_routes.jsonl",
    "val_escape": DATASET_DIR / "val_escape.jsonl",
}

DATASET_NAMES = tuple(DATASET_PATHS)
DEFAULT_DATASET = "heldout"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_dataset(name: str) -> list[dict]:
    """Load a dataset by name.

    For 'val', the source file is exemplar format {route, lang, utterance} which
    must be converted to the eval_router.py form {utterance, lang, tool, args}
    using route_schema.resolve_expected_route inverse mapping. Other datasets are
    read directly as JSONL.
    """
    if name == "val":
        return _load_val()
    return read_jsonl(DATASET_PATHS[name])


def _load_val() -> list[dict]:
    """Load val_routes.jsonl and convert from exemplar format to eval_router format.

    The val file has {route, lang, utterance} rows. eval_router.py's
    load_labelled_utterances calls resolve_expected_route(row["tool"], row["args"])
    to get expected_route. We need to map the route string back to tool + args.

    A route like "seek_time:start" means tool="seek_time", args={"position": "start"}.
    A plain route like "list_objects" means tool="list_objects", args={}.
    """
    from route_schema import ROUTE_DIMENSIONS, build_routes

    # Build a mapping from route_id -> (tool_name, enum_args_dict)
    routes = build_routes()
    route_to_parts: dict[str, tuple[str, dict]] = {}
    for route in routes:
        enum_args = dict(route.enum_args)
        route_to_parts[route.route_id] = (route.tool_name, enum_args)

    rows = read_jsonl(DATASET_PATHS["val"])
    result = []
    for row in rows:
        route_id = row["route"]
        tool_name, enum_args = route_to_parts.get(route_id, (route_id, {}))
        result.append({
            "utterance": row["utterance"],
            "lang": row["lang"],
            "tool": tool_name,
            "args": enum_args,
        })
    return result
