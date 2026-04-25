from __future__ import annotations

from dataclasses import dataclass

from .stubs import pb2, pb2_grpc
from .client import open_channel, call_with_retry, build_call_metadata
from .config import ServerConfig


@dataclass(frozen=True)
class AutoRigInput:
    glb_data: bytes
    num_sample_points: int = 65536


@dataclass(frozen=True)
class SkeletonJointInfo:
    name: str
    head: tuple[float, float, float]
    tail: tuple[float, float, float]
    parent_index: int


@dataclass(frozen=True)
class AutoRigResult:
    rigged_glb_data: bytes
    joint_count: int
    bone_count: int
    generation_time_ms: float
    skeleton_joints: tuple[SkeletonJointInfo, ...]


@dataclass(frozen=True)
class AutoRigStatus:
    ready: bool
    model_name: str
    gpu_memory_mb: int


class AutoRigClient:
    def __init__(self, config: ServerConfig) -> None:
        self._config = config

    def generate_rig(self, input_data: AutoRigInput) -> AutoRigResult:
        request = pb2.RiggingRequest(
            glb_data=input_data.glb_data,
            params=pb2.RiggingParams(num_sample_points=input_data.num_sample_points),
            model_type=pb2.RIGGING_UNIRIG,
        )

        with open_channel(self._config) as channel:
            stub = pb2_grpc.AutoRiggingServiceStub(channel)
            response = call_with_retry(
                lambda: stub.GenerateRig(
                    request,
                    timeout=self._config.deadline_seconds,
                    metadata=build_call_metadata(self._config),
                ),
                config=self._config,
                operation_name="AutoRiggingService.GenerateRig",
            )

        return _convert_rigging_response(response)

    def get_status(self) -> AutoRigStatus:
        with open_channel(self._config) as channel:
            stub = pb2_grpc.AutoRiggingServiceStub(channel)
            response = call_with_retry(
                lambda: stub.GetRiggingStatus(
                    pb2.RiggingStatusRequest(),
                    timeout=10.0,
                    metadata=build_call_metadata(self._config),
                ),
                config=self._config,
                operation_name="AutoRiggingService.GetRiggingStatus",
            )

        return AutoRigStatus(
            ready=response.ready,
            model_name=response.model_name,
            gpu_memory_mb=response.gpu_memory_mb,
        )


def _convert_rigging_response(response: pb2.RiggingResponse) -> AutoRigResult:
    joints = tuple(
        SkeletonJointInfo(
            name=joint.name,
            head=(joint.x, joint.y, joint.z),
            tail=(joint.tail_x, joint.tail_y, joint.tail_z),
            parent_index=joint.parent_index,
        )
        for joint in response.skeleton_joints
    )

    if response.HasField("metadata"):
        meta = response.metadata
        joint_count = meta.joint_count
        bone_count = meta.bone_count
        generation_time_ms = meta.generation_time_ms
    else:
        joint_count = 0
        bone_count = 0
        generation_time_ms = 0.0

    return AutoRigResult(
        rigged_glb_data=response.rigged_glb_data,
        joint_count=joint_count,
        bone_count=bone_count,
        generation_time_ms=generation_time_ms,
        skeleton_joints=joints,
    )
