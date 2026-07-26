"""Machine-specific paths, resolved from the file that owns them.

`.claude/local/paths.md` is untracked, so absolute paths stay out of the
repository while every script still points at the same directories. Hard-coding
them here would put a home directory into version control.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PATHS_FILE = REPO_ROOT / ".claude" / "local" / "paths.md"

PATH_ENTRY = re.compile(r"^-\s*(\w+)\s*=\s*(\S+)", re.MULTILINE)


def read_local_paths() -> dict[str, str]:
    if not PATHS_FILE.exists():
        raise FileNotFoundError(f"{PATHS_FILE} is required to resolve machine paths")
    return dict(PATH_ENTRY.findall(PATHS_FILE.read_text(encoding="utf-8")))


def find_model_dir(model_name: str) -> str:
    storage = read_local_paths().get("ModelStoragePath")
    if storage is None:
        raise KeyError("ModelStoragePath is not defined in .claude/local/paths.md")
    return str(Path(storage) / model_name)
