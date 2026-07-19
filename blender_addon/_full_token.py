"""Full-token resolution across build modes.

Only mode B ships a token source (``telemetry``, token-refresh for feedback
senders). Mode A ships none and degrades to ctx32; mode C (private) ships
none either -- its wheel path bypasses the token gate entirely via
``CAPS.curve_copilot_mode == "private"``.
"""
from __future__ import annotations

try:
    from . import telemetry
except ImportError:
    telemetry = None


def resolve_full_token() -> str | None:
    if telemetry is not None:
        return telemetry.resolve_full_token()
    return None
