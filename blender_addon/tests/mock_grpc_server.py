from __future__ import annotations

import time
from concurrent import futures
from dataclasses import dataclass, field

import grpc

from blender_addon.grpc_client.stubs import pb2, pb2_grpc


@dataclass
class MockBehavior:
    """Per-service behavior knobs injected into mock services."""
    return_status: grpc.StatusCode = grpc.StatusCode.OK
    sleep_seconds: float = 0.0
    fail_first_n_calls: int = 0
    captured_metadata: list[tuple[str, str]] = field(default_factory=list)
    captured_glb_size: int = 0
    captured_num_sample_points: int = 0
    captured_prompt: str = ""
    call_count: int = 0


def _record_call(behavior: MockBehavior, context: grpc.ServicerContext) -> None:
    behavior.call_count += 1
    behavior.captured_metadata = [
        (key, value) for key, value in context.invocation_metadata()
    ]


def _maybe_abort(behavior: MockBehavior, context: grpc.ServicerContext) -> bool:
    """Apply sleep / failure injection. Return True if the call should produce a real response."""
    if behavior.sleep_seconds > 0:
        time.sleep(behavior.sleep_seconds)
    if behavior.call_count <= behavior.fail_first_n_calls:
        context.abort(grpc.StatusCode.UNAVAILABLE, "mock transient failure")
        return False
    if behavior.return_status != grpc.StatusCode.OK:
        context.abort(behavior.return_status, "mock injected error")
        return False
    return True


class MockAutoRiggingService(pb2_grpc.AutoRiggingServiceServicer):
    def __init__(self, behavior: MockBehavior) -> None:
        self.behavior = behavior

    def GenerateRig(self, request, context):
        _record_call(self.behavior, context)
        self.behavior.captured_glb_size = len(request.glb_data)
        if request.HasField("params"):
            self.behavior.captured_num_sample_points = request.params.num_sample_points
        if not _maybe_abort(self.behavior, context):
            return pb2.RiggingResponse()
        return _ok_rigging_response()

    def GetRiggingStatus(self, request, context):
        _record_call(self.behavior, context)
        if not _maybe_abort(self.behavior, context):
            return pb2.RiggingStatusResponse()
        return pb2.RiggingStatusResponse(
            ready=True, model_name="MockUniRig", gpu_memory_mb=8192,
        )


class MockMeshGenerationService(pb2_grpc.MeshGenerationServiceServicer):
    def __init__(self, behavior: MockBehavior) -> None:
        self.behavior = behavior

    def GenerateMesh(self, request, context):
        _record_call(self.behavior, context)
        self.behavior.captured_prompt = request.prompt
        if not _maybe_abort(self.behavior, context):
            return pb2.MeshResponse()
        return _ok_mesh_response()

    def GetMeshServiceStatus(self, request, context):
        _record_call(self.behavior, context)
        if not _maybe_abort(self.behavior, context):
            return pb2.MeshStatusResponse()
        return pb2.MeshStatusResponse(
            ready=True, t2i_model="MockSDXL", i2m_model="MockTRELLIS", gpu_memory_mb=16384,
        )


class MockTextToMotionService(pb2_grpc.TextToMotionServiceServicer):
    def __init__(self, behavior: MockBehavior) -> None:
        self.behavior = behavior

    def GenerateMotion(self, request, context):
        _record_call(self.behavior, context)
        self.behavior.captured_prompt = request.prompt
        if not _maybe_abort(self.behavior, context):
            return pb2.MotionResponse()
        return _ok_motion_response()

    def GetServerStatus(self, request, context):
        _record_call(self.behavior, context)
        if not _maybe_abort(self.behavior, context):
            return pb2.StatusResponse()
        return pb2.StatusResponse(
            ready=True, active_model="MockLightT2M", gpu_memory_mb=4096,
        )


def _ok_rigging_response() -> pb2.RiggingResponse:
    return pb2.RiggingResponse(
        rigged_glb_data=b"mock_rigged_glb",
        metadata=pb2.RiggingMetadata(
            joint_count=22, bone_count=21, generation_time_ms=42.0,
        ),
        skeleton_joints=[
            pb2.SkeletonJoint(
                name="hips",
                x=0.0, y=1.0, z=0.0,
                tail_x=0.0, tail_y=1.1, tail_z=0.0,
                parent_index=-1,
            ),
        ],
    )


def _ok_mesh_response() -> pb2.MeshResponse:
    return pb2.MeshResponse(
        glb_data=b"mock_mesh_glb",
        metadata=pb2.MeshMetadata(
            vertex_count=100, face_count=50, generation_time_ms=123.0,
            intermediate_image_png=b"",
        ),
    )


def _ok_motion_response() -> pb2.MotionResponse:
    return pb2.MotionResponse(
        curves=[
            pb2.AnimationCurve(
                bone_name="hips",
                property_type=pb2.ROTATION_X,
                keyframes=[
                    pb2.CurveKeyframe(
                        time=0.0, value=0.0,
                        tangent_in_dt=0.0, tangent_in_dv=0.0,
                        tangent_out_dt=0.1, tangent_out_dv=0.0,
                        interpolation=pb2.BEZIER,
                    ),
                ],
            ),
        ],
        generation_time_ms=78.0,
        model_used="MockLightT2M",
    )


@dataclass
class MockServerHandle:
    server: grpc.Server
    port: int
    auto_rig_behavior: MockBehavior
    mesh_behavior: MockBehavior
    motion_behavior: MockBehavior

    def stop(self, grace_seconds: float = 0.5) -> None:
        self.server.stop(grace=grace_seconds).wait()


def start_mock_server(max_workers: int = 4) -> MockServerHandle:
    auto_rig_behavior = MockBehavior()
    mesh_behavior = MockBehavior()
    motion_behavior = MockBehavior()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb2_grpc.add_AutoRiggingServiceServicer_to_server(
        MockAutoRiggingService(auto_rig_behavior), server,
    )
    pb2_grpc.add_MeshGenerationServiceServicer_to_server(
        MockMeshGenerationService(mesh_behavior), server,
    )
    pb2_grpc.add_TextToMotionServiceServicer_to_server(
        MockTextToMotionService(motion_behavior), server,
    )

    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    return MockServerHandle(
        server=server,
        port=port,
        auto_rig_behavior=auto_rig_behavior,
        mesh_behavior=mesh_behavior,
        motion_behavior=motion_behavior,
    )
