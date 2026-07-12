from __future__ import annotations

import pytest
import grpc

from blender_addon.grpc_client import (
    AutoRigClient,
    AutoRigInput,
    GrpcClientError,
    GrpcConnectionError,
    GrpcServerError,
    GrpcTimeoutError,
    RetryPolicy,
    ServerConfig,
)
from blender_addon.tests.mock_grpc_server import MockServerHandle


def _make_input() -> AutoRigInput:
    return AutoRigInput(glb_data=b"\x00" * 16, num_sample_points=1024)


def test_generate_rig_succeeds_against_mock(mock_server: MockServerHandle, server_config: ServerConfig) -> None:
    client = AutoRigClient(server_config)
    result = client.generate_rig(_make_input())

    assert result.bone_count == 21
    assert result.joint_count == 22
    assert len(result.skeleton_joints) == 1
    assert mock_server.auto_rig_behavior.captured_glb_size == 16
    assert mock_server.auto_rig_behavior.captured_num_sample_points == 1024


def test_invalid_argument_does_not_retry(mock_server: MockServerHandle, server_config: ServerConfig) -> None:
    mock_server.auto_rig_behavior.return_status = grpc.StatusCode.INVALID_ARGUMENT
    client = AutoRigClient(server_config)

    with pytest.raises(GrpcServerError) as info:
        client.generate_rig(_make_input())

    assert "INVALID_ARGUMENT" in info.value.code
    assert mock_server.auto_rig_behavior.call_count == 1


def test_unavailable_retries_then_succeeds(mock_server: MockServerHandle, server_config: ServerConfig) -> None:
    mock_server.auto_rig_behavior.fail_first_n_calls = 2
    client = AutoRigClient(server_config)

    result = client.generate_rig(_make_input())

    assert result.bone_count == 21
    assert mock_server.auto_rig_behavior.call_count == 3


def test_retry_exhaustion_raises_connection_error(mock_server: MockServerHandle, server_config: ServerConfig) -> None:
    mock_server.auto_rig_behavior.fail_first_n_calls = 99
    client = AutoRigClient(server_config)

    with pytest.raises(GrpcConnectionError):
        client.generate_rig(_make_input())

    assert mock_server.auto_rig_behavior.call_count == server_config.retry.max_attempts


def test_deadline_exceeded_raises_timeout(mock_server: MockServerHandle) -> None:
    config = ServerConfig(
        host="127.0.0.1",
        port=mock_server.port,
        deadline_seconds=0.1,
        retry=RetryPolicy(
            max_attempts=1,
            initial_backoff_seconds=0.01,
            backoff_multiplier=1.0,
            max_backoff_seconds=0.05,
        ),
    )
    mock_server.auto_rig_behavior.sleep_seconds = 1.0
    client = AutoRigClient(config)

    with pytest.raises(GrpcTimeoutError):
        client.generate_rig(_make_input())


def test_unreachable_server_raises_client_error(fast_retry: RetryPolicy) -> None:
    """Connecting to a closed port should surface as a client-layer error.

    Real "unreachable" semantics depend on TCP timing: a fast RST may produce
    UNAVAILABLE while a stuck SYN may exhaust the deadline first. Either path
    must reach the operator layer through the public exception hierarchy, so
    the test asserts on the base class.
    """
    config = ServerConfig(
        host="127.0.0.1",
        port=1,
        deadline_seconds=0.5,
        retry=fast_retry,
    )
    client = AutoRigClient(config)

    with pytest.raises(GrpcClientError):
        client.generate_rig(_make_input())


def test_auth_metadata_is_propagated(mock_server: MockServerHandle, fast_retry: RetryPolicy) -> None:
    config = ServerConfig(
        host="127.0.0.1",
        port=mock_server.port,
        deadline_seconds=2.0,
        retry=fast_retry,
        auth_metadata=(("authorization", "Bearer test-token"),),
    )
    client = AutoRigClient(config)

    client.generate_rig(_make_input())

    captured = dict(mock_server.auto_rig_behavior.captured_metadata)
    assert captured.get("authorization") == "Bearer test-token"
