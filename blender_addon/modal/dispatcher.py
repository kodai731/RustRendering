"""AsyncDispatcher - run a gRPC call on a worker thread, deliver via queue.

Used by Tier A Modal Operators (auto_rig / text_to_mesh / text_to_motion). The
worker is forbidden from touching ``bpy.*`` directly; bpy mutation happens in
the Modal Operator's Timer tick on the main thread.
"""
from __future__ import annotations

import os
import queue
import threading
from typing import Any, Callable, Optional, Tuple


HEADLESS_ENV_VAR = "THYLLORE_HEADLESS"


def is_headless() -> bool:
    """Return True when the addon is running under a headless test harness.

    Modal operators bypass their worker-thread / Timer pipeline in this mode
    and call the gRPC / PyO3 entry point synchronously inside ``execute``,
    because ``bpy.app.timers`` does not tick under ``blender --background``.
    """
    return os.environ.get(HEADLESS_ENV_VAR) == "1"


def run_sync_if_headless(
    target: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Optional[Tuple[str, Any]]:
    """Synchronous fallback for headless mode.

    Returns the same ``("ok", result)`` / ``("err", exc)`` tuple shape that
    ``AsyncDispatcher.poll()`` would yield, so call sites can share a single
    result-handling branch. Returns ``None`` when not in headless mode, so the
    caller can fall through to the normal Modal path.
    """
    if not is_headless():
        return None
    try:
        result = target(*args, **kwargs)
        return ("ok", result)
    except BaseException as exc:  # noqa: BLE001
        return ("err", exc)


class AsyncDispatcher:
    """Run a callable on a worker thread; expose its result via queue.

    Instantiated per-Operator-call, not shared. ``cancel()`` is idempotent and
    safe to call from ``Operator.cancel()`` or the Modal exit branch.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._cancelled: bool = False

    def start(
        self,
        target: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Spawn the worker thread.

        Raises ``RuntimeError`` if a worker is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("AsyncDispatcher already running")

        self._cancelled = False
        self._queue = queue.Queue()

        def _runner() -> None:
            try:
                result = target(*args, **kwargs)
                if not self._cancelled:
                    self._queue.put(("ok", result))
            except BaseException as exc:  # noqa: BLE001
                if not self._cancelled:
                    self._queue.put(("err", exc))

        self._thread = threading.Thread(
            target=_runner,
            name="thyllore-async-dispatcher",
            daemon=True,
        )
        self._thread.start()

    def poll(self) -> Optional[Tuple[str, Any]]:
        """Non-blocking peek. Returns ``None`` if not ready."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def cancel(self) -> None:
        """Mark cancelled and join the worker thread (5s timeout).

        Subsequent ``poll()`` calls drain whatever the worker put before the
        cancellation flag was observed; callers should ignore them. Idempotent.
        """
        self._cancelled = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                print("[Thyllore] WARNING: dispatcher worker did not exit in 5s")
            self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
