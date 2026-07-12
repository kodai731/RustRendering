"""Full-token resolution across build modes.

Exactly one token source ships per build: ``telemetry`` (mode B, token-refresh
for feedback senders) or ``license_client`` (mode C, seat-bound licensing).
Mode A ships neither, so this always resolves to None and the wheel degrades
to ctx32.
"""
from __future__ import annotations

try:
    from . import telemetry
except ImportError:
    telemetry = None

try:
    from . import license_client
except ImportError:
    license_client = None


def resolve_full_token() -> str | None:
    if telemetry is not None:
        return telemetry.resolve_full_token()
    if license_client is not None:
        return license_client.resolve_full_token()
    return None
