from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


_DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff_seconds: float = 0.5
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 8.0


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 50051
    use_tls: bool = False
    deadline_seconds: float = 120.0
    max_send_message_bytes: int = _DEFAULT_MAX_MESSAGE_BYTES
    max_receive_message_bytes: int = _DEFAULT_MAX_MESSAGE_BYTES
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    auth_metadata: Optional[tuple[tuple[str, str], ...]] = None

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    def channel_options(self) -> list[tuple[str, int]]:
        return [
            ("grpc.max_send_message_length", self.max_send_message_bytes),
            ("grpc.max_receive_message_length", self.max_receive_message_bytes),
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
            ("grpc.keepalive_permit_without_calls", 1),
        ]
