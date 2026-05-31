"""Developer debug logging for the Thyllore Animation addon.

Caller layer: the actual file output lives in the Rust wheel
(``thyllore_ml_core.debug_log`` / ``debug_log_init``), compiled only under the
wheel's ``debug-log`` feature. Production wheels omit those functions, so
``_rust_log`` is ``None`` here and logging is impossible regardless of any env
var or config — the build-time switch is the Rust feature, mirroring the
``#[cfg(feature = "python")]`` pattern used elsewhere.

This module only does the Blender-side glue: decide whether to log, resolve the
log file path, and format messages before handing them to Rust.

Activation requires the debug-log wheel AND one of:
1. ``THYLLORE_DEBUG`` environment variable is truthy — dev override.
2. A ``_debug_config.py`` baked into the addon by ``build_blender_addon.sh
   --debug`` sets ``DEBUG_BUILD = True``.

Log file path (first match wins): ``THYLLORE_BLENDER_LOG`` env, baked
``LOG_PATH``, or ``<repo_root>/log/log_blender.log`` discovered from the source
tree.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

try:
    import thyllore_ml_core as _tml  # type: ignore

    _rust_log_init = getattr(_tml, "debug_log_init", None)
    _rust_log = getattr(_tml, "debug_log", None)
except ImportError:
    _rust_log_init = None
    _rust_log = None

_TRUTHY = {"1", "true", "yes", "on"}


def _baked_config() -> Tuple[bool, str]:
    try:
        from . import _debug_config as config
    except ImportError:
        return False, ""
    return bool(getattr(config, "DEBUG_BUILD", False)), str(getattr(config, "LOG_PATH", ""))


def is_debug_enabled() -> bool:
    # The Rust output methods are the hard build-time gate: a production wheel
    # has none, so logging can never activate.
    if _rust_log is None or _rust_log_init is None:
        return False
    if os.environ.get("THYLLORE_DEBUG", "").strip().lower() in _TRUTHY:
        return True
    baked_on, _ = _baked_config()
    return baked_on


def _discover_repo_log_path() -> Optional[Path]:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Cargo.toml").is_file() and (parent / "log").is_dir():
            return parent / "log" / "log_blender.log"
    return None


def _resolve_log_path() -> Optional[Path]:
    env = os.environ.get("THYLLORE_BLENDER_LOG", "").strip()
    if env:
        return Path(env)

    _, baked = _baked_config()
    if baked:
        return Path(baked)

    return _discover_repo_log_path()


class _RustLogger:
    """Formats messages and forwards them to the Rust file logger."""

    def info(self, message: str, *args) -> None:
        _rust_log(message % args if args else message)

    def exception(self, message: str) -> None:
        import traceback

        _rust_log(f"{message}\n{traceback.format_exc()}")


_logger: Optional[_RustLogger] = None
_initialized = False


def get_logger() -> Optional[_RustLogger]:
    """Return a logger that writes via Rust, or ``None`` when debug is off."""
    global _logger, _initialized
    if _initialized:
        return _logger

    _initialized = True
    if not is_debug_enabled():
        return None

    log_path = _resolve_log_path()
    if log_path is None:
        print("[Thyllore] debug mode on but no writable log path resolved")
        return None

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _rust_log_init(str(log_path))
    except OSError as e:
        print(f"[Thyllore] failed to open debug log {log_path}: {e}")
        return None

    _logger = _RustLogger()
    _logger.info("debug log opened at %s", str(log_path))
    print(f"[Thyllore] debug log -> {log_path}")
    return _logger
