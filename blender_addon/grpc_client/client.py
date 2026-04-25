from __future__ import annotations

import contextlib
import logging
import time
from typing import Callable, Iterator, TypeVar

import grpc

from .config import ServerConfig


log = logging.getLogger(__name__)


class GrpcClientError(Exception):
    """Base for all gRPC client errors visible to the operator layer."""


class GrpcConnectionError(GrpcClientError):
    """Server unreachable (UNAVAILABLE / CANCELLED / DNS failure)."""


class GrpcTimeoutError(GrpcClientError):
    """Deadline exceeded."""


class GrpcServerError(GrpcClientError):
    """Server returned a non-OK status (INVALID_ARGUMENT, INTERNAL, etc.)."""

    def __init__(self, code: str, details: str) -> None:
        super().__init__(f"{code}: {details}")
        self.code = code
        self.details = details


_RETRYABLE_CODES = frozenset({
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
})


@contextlib.contextmanager
def open_channel(config: ServerConfig) -> Iterator[grpc.Channel]:
    if config.use_tls:
        creds = grpc.ssl_channel_credentials()
        channel = grpc.secure_channel(
            config.endpoint, creds, options=config.channel_options()
        )
    else:
        channel = grpc.insecure_channel(
            config.endpoint, options=config.channel_options()
        )
    try:
        yield channel
    finally:
        channel.close()


def build_call_metadata(config: ServerConfig) -> list[tuple[str, str]]:
    if config.auth_metadata is None:
        return []
    return list(config.auth_metadata)


T = TypeVar("T")


def call_with_retry(
    rpc: Callable[[], T],
    config: ServerConfig,
    operation_name: str,
) -> T:
    """Execute `rpc()` with the retry policy and convert gRPC errors to client errors."""
    backoff = config.retry.initial_backoff_seconds
    last_error: Exception | None = None

    for attempt in range(1, config.retry.max_attempts + 1):
        try:
            return rpc()
        except grpc.RpcError as err:
            code = _extract_status_code(err)
            details = _extract_status_details(err)

            should_retry = (
                code in _RETRYABLE_CODES
                and attempt < config.retry.max_attempts
            )

            if not should_retry:
                raise _convert_to_client_error(code, operation_name, details) from err

            log.warning(
                "%s attempt %d/%d failed (%s), retrying in %.2fs",
                operation_name,
                attempt,
                config.retry.max_attempts,
                code,
                backoff,
            )
            time.sleep(backoff)
            backoff = min(
                backoff * config.retry.backoff_multiplier,
                config.retry.max_backoff_seconds,
            )
            last_error = err

    assert last_error is not None
    raise GrpcConnectionError(f"{operation_name}: exhausted retries") from last_error


def _extract_status_code(err: grpc.RpcError) -> grpc.StatusCode:
    code_fn = getattr(err, "code", None)
    if callable(code_fn):
        try:
            value = code_fn()
            if isinstance(value, grpc.StatusCode):
                return value
        except Exception:
            pass
    return grpc.StatusCode.UNKNOWN


def _extract_status_details(err: grpc.RpcError) -> str:
    details_fn = getattr(err, "details", None)
    if callable(details_fn):
        try:
            value = details_fn()
            if value is not None:
                return str(value)
        except Exception:
            pass
    return str(err)


def _convert_to_client_error(
    code: grpc.StatusCode,
    operation_name: str,
    details: str,
) -> GrpcClientError:
    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return GrpcTimeoutError(f"{operation_name}: {details}")
    if code == grpc.StatusCode.UNAVAILABLE:
        return GrpcConnectionError(f"{operation_name}: {details}")
    return GrpcServerError(str(code), f"{operation_name}: {details}")
