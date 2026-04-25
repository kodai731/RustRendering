from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .stubs import pb2, pb2_grpc
from .client import open_channel, call_with_retry, build_call_metadata
from .config import ServerConfig
from . import _enums


@dataclass(frozen=True)
class BoneMappingInput:
    source_joint_index: int
    target_bone_name: str


@dataclass(frozen=True)
class MotionInput:
    prompt: str
    duration_seconds: float
    target_fps: int = 30
    skeleton_type: str = "SMPL_22"
    bone_mappings: tuple[BoneMappingInput, ...] = field(default_factory=tuple)
    glb_data: Optional[bytes] = None
    skeleton_cache_id: str = ""
    internal_use_only: bool = False


@dataclass(frozen=True)
class CurveKeyframe:
    time: float
    value: float
    tangent_in_dt: float
    tangent_in_dv: float
    tangent_out_dt: float
    tangent_out_dv: float
    interpolation: str


@dataclass(frozen=True)
class AnimationCurve:
    bone_name: str
    property_type: str
    keyframes: tuple[CurveKeyframe, ...]


@dataclass(frozen=True)
class MotionResult:
    curves: tuple[AnimationCurve, ...]
    generation_time_ms: float
    model_used: str


@dataclass(frozen=True)
class MotionStatus:
    ready: bool
    active_model: str
    gpu_memory_mb: int


class MotionClient:
    def __init__(self, config: ServerConfig) -> None:
        self._config = config

    def generate_motion(self, input_data: MotionInput) -> MotionResult:
        request = _build_motion_request(input_data)

        with open_channel(self._config) as channel:
            stub = pb2_grpc.TextToMotionServiceStub(channel)
            response = call_with_retry(
                lambda: stub.GenerateMotion(
                    request,
                    timeout=self._config.deadline_seconds,
                    metadata=build_call_metadata(self._config),
                ),
                config=self._config,
                operation_name="TextToMotionService.GenerateMotion",
            )

        return _convert_motion_response(response)

    def get_status(self) -> MotionStatus:
        with open_channel(self._config) as channel:
            stub = pb2_grpc.TextToMotionServiceStub(channel)
            response = call_with_retry(
                lambda: stub.GetServerStatus(
                    pb2.StatusRequest(),
                    timeout=10.0,
                    metadata=build_call_metadata(self._config),
                ),
                config=self._config,
                operation_name="TextToMotionService.GetServerStatus",
            )

        return MotionStatus(
            ready=response.ready,
            active_model=response.active_model,
            gpu_memory_mb=response.gpu_memory_mb,
        )


def _build_motion_request(input_data: MotionInput) -> pb2.MotionRequest:
    bone_mappings = [
        pb2.BoneMapping(
            source_joint_index=mapping.source_joint_index,
            target_bone_name=mapping.target_bone_name,
        )
        for mapping in input_data.bone_mappings
    ]

    if input_data.glb_data is not None or input_data.skeleton_cache_id:
        glb_skeleton = pb2.GlbSkeletonSpec(
            glb_data=input_data.glb_data or b"",
            skeleton_cache_id=input_data.skeleton_cache_id,
        )
    else:
        glb_skeleton = None

    request = pb2.MotionRequest(
        prompt=input_data.prompt,
        duration_seconds=input_data.duration_seconds,
        target_fps=input_data.target_fps,
        skeleton_type=_enums.skeleton_type_int(input_data.skeleton_type),
        bone_mappings=bone_mappings,
        internal_use_only=input_data.internal_use_only,
    )
    if glb_skeleton is not None:
        request.glb_skeleton.CopyFrom(glb_skeleton)
    return request


def _convert_motion_response(response: pb2.MotionResponse) -> MotionResult:
    curves = tuple(
        AnimationCurve(
            bone_name=curve.bone_name,
            property_type=_enums.property_type_str(curve.property_type),
            keyframes=tuple(
                CurveKeyframe(
                    time=kf.time,
                    value=kf.value,
                    tangent_in_dt=kf.tangent_in_dt,
                    tangent_in_dv=kf.tangent_in_dv,
                    tangent_out_dt=kf.tangent_out_dt,
                    tangent_out_dv=kf.tangent_out_dv,
                    interpolation=_enums.interpolation_str(kf.interpolation),
                )
                for kf in curve.keyframes
            ),
        )
        for curve in response.curves
    )

    return MotionResult(
        curves=curves,
        generation_time_ms=response.generation_time_ms,
        model_used=response.model_used,
    )
