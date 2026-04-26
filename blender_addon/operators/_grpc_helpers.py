"""Shared helpers for Tier A (gRPC) operators.

Kept private (leading underscore) to make explicit that these are not part of
the addon's stable surface. Operators import them via relative import only.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from ..grpc_client import (
    GrpcClientError,
    GrpcConnectionError,
    GrpcServerError,
    GrpcTimeoutError,
    ServerConfig,
)


def build_server_config_from_preferences() -> ServerConfig:
    """Construct a ServerConfig from the current AddonPreferences."""
    from .. import preferences as prefs_module

    prefs = prefs_module.get_preferences()

    auth_metadata = None
    if prefs.license_key:
        auth_metadata = (("authorization", f"Bearer {prefs.license_key}"),)

    return ServerConfig(
        host=prefs.server_host,
        port=prefs.server_port,
        use_tls=prefs.use_tls,
        deadline_seconds=prefs.deadline_seconds,
        auth_metadata=auth_metadata,
    )


def grpc_error_message(exc: BaseException) -> str:
    """Convert a known grpc_client exception into a UI-friendly string."""
    if isinstance(exc, GrpcConnectionError):
        return f"Cannot connect to server: {exc}"
    if isinstance(exc, GrpcTimeoutError):
        return f"Server timed out: {exc}"
    if isinstance(exc, GrpcServerError):
        return f"Server returned error: {exc}"
    if isinstance(exc, GrpcClientError):
        return f"Client error: {exc}"
    return f"Unexpected error: {exc}"


def write_temp_glb(glb_bytes: bytes, prefix: str) -> Path:
    """Write GLB bytes to a temp file and return the path. Caller deletes."""
    fd = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".glb", delete=False)
    try:
        fd.write(glb_bytes)
    finally:
        fd.close()
    return Path(fd.name)
