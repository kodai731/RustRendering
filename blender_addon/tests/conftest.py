from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# grpcio wheels are cp311; on other interpreters (or without extracted wheels)
# the gRPC fixtures skip instead of breaking unrelated test collection.
try:
    from blender_addon.grpc_client.config import RetryPolicy, ServerConfig
    from blender_addon.tests.mock_grpc_server import MockServerHandle, start_mock_server

    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False


@pytest.fixture
def mock_server():
    if not _GRPC_AVAILABLE:
        pytest.skip("grpcio unavailable on this interpreter")
    handle = start_mock_server()
    try:
        yield handle
    finally:
        handle.stop()


@pytest.fixture
def fast_retry():
    if not _GRPC_AVAILABLE:
        pytest.skip("grpcio unavailable on this interpreter")
    return RetryPolicy(
        max_attempts=3,
        initial_backoff_seconds=0.01,
        backoff_multiplier=1.5,
        max_backoff_seconds=0.1,
    )


@pytest.fixture
def server_config(mock_server, fast_retry):
    return ServerConfig(
        host="127.0.0.1",
        port=mock_server.port,
        deadline_seconds=2.0,
        retry=fast_retry,
    )
