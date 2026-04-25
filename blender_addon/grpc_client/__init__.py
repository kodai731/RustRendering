"""gRPC client layer for the ThylloreAnimation Blender addon (pure Python, no bpy dependency)."""

from .config import RetryPolicy, ServerConfig
from .client import (
    GrpcClientError,
    GrpcConnectionError,
    GrpcServerError,
    GrpcTimeoutError,
)
from .auto_rig import (
    AutoRigClient,
    AutoRigInput,
    AutoRigResult,
    AutoRigStatus,
    SkeletonJointInfo,
)
from .mesh import MeshClient, MeshInput, MeshResult, MeshStatus
from .motion import (
    AnimationCurve,
    BoneMappingInput,
    CurveKeyframe,
    MotionClient,
    MotionInput,
    MotionResult,
    MotionStatus,
)

__all__ = [
    "RetryPolicy",
    "ServerConfig",
    "GrpcClientError",
    "GrpcConnectionError",
    "GrpcServerError",
    "GrpcTimeoutError",
    "AutoRigClient",
    "AutoRigInput",
    "AutoRigResult",
    "AutoRigStatus",
    "SkeletonJointInfo",
    "MeshClient",
    "MeshInput",
    "MeshResult",
    "MeshStatus",
    "AnimationCurve",
    "BoneMappingInput",
    "CurveKeyframe",
    "MotionClient",
    "MotionInput",
    "MotionResult",
    "MotionStatus",
]
