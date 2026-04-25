from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blender_addon.grpc_client.config import RetryPolicy, ServerConfig
from blender_addon.tests.mock_grpc_server import MockServerHandle, start_mock_server


@pytest.fixture
def mock_server() -> MockServerHandle:
    handle = start_mock_server()
    try:
        yield handle
    finally:
        handle.stop()


@pytest.fixture
def fast_retry() -> RetryPolicy:
    return RetryPolicy(
        max_attempts=3,
        initial_backoff_seconds=0.01,
        backoff_multiplier=1.5,
        max_backoff_seconds=0.1,
    )


@pytest.fixture
def server_config(mock_server: MockServerHandle, fast_retry: RetryPolicy) -> ServerConfig:
    return ServerConfig(
        host="127.0.0.1",
        port=mock_server.port,
        deadline_seconds=2.0,
        retry=fast_retry,
    )
