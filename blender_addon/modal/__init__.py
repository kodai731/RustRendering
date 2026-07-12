"""Modal Operator support for non-blocking gRPC calls."""
from .dispatcher import AsyncDispatcher, is_headless, run_sync_if_headless

__all__ = ["AsyncDispatcher", "is_headless", "run_sync_if_headless"]
