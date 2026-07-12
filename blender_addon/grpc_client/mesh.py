from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .stubs import pb2, pb2_grpc
from .client import open_channel, call_with_retry, build_call_metadata
from .config import ServerConfig
from . import _enums


@dataclass(frozen=True)
class MeshInput:
    prompt: str
    target_faces: int = 30000
    seed: int = 42
    image_size: int = 768
    image_inference_steps: int = 30
    input_mode: str = "TEXT_TO_MESH"
    input_image_png: Optional[bytes] = None
    model_type: str = "TRELLIS"
    t2i_model_type: str = "T2I_SERVER_DEFAULT"


@dataclass(frozen=True)
class MeshResult:
    glb_data: bytes
    vertex_count: int
    face_count: int
    generation_time_ms: float
    intermediate_image_png: Optional[bytes]


@dataclass(frozen=True)
class MeshStatus:
    ready: bool
    t2i_model: str
    i2m_model: str
    gpu_memory_mb: int


class MeshClient:
    def __init__(self, config: ServerConfig) -> None:
        self._config = config

    def generate_mesh(self, input_data: MeshInput) -> MeshResult:
        request = pb2.MeshRequest(
            prompt=input_data.prompt,
            params=pb2.MeshGenerationParams(
                target_faces=input_data.target_faces,
                seed=input_data.seed,
                image_size=input_data.image_size,
                image_inference_steps=input_data.image_inference_steps,
            ),
            input_image_png=input_data.input_image_png or b"",
            input_mode=_enums.mesh_input_mode_int(input_data.input_mode),
            model_type=_enums.mesh_model_type_int(input_data.model_type),
            t2i_model_type=_enums.text_to_image_model_type_int(input_data.t2i_model_type),
        )

        with open_channel(self._config) as channel:
            stub = pb2_grpc.MeshGenerationServiceStub(channel)
            response = call_with_retry(
                lambda: stub.GenerateMesh(
                    request,
                    timeout=self._config.deadline_seconds,
                    metadata=build_call_metadata(self._config),
                ),
                config=self._config,
                operation_name="MeshGenerationService.GenerateMesh",
            )

        return _convert_mesh_response(response)

    def get_status(self) -> MeshStatus:
        with open_channel(self._config) as channel:
            stub = pb2_grpc.MeshGenerationServiceStub(channel)
            response = call_with_retry(
                lambda: stub.GetMeshServiceStatus(
                    pb2.MeshStatusRequest(),
                    timeout=10.0,
                    metadata=build_call_metadata(self._config),
                ),
                config=self._config,
                operation_name="MeshGenerationService.GetMeshServiceStatus",
            )

        return MeshStatus(
            ready=response.ready,
            t2i_model=response.t2i_model,
            i2m_model=response.i2m_model,
            gpu_memory_mb=response.gpu_memory_mb,
        )


def _convert_mesh_response(response: pb2.MeshResponse) -> MeshResult:
    if response.HasField("metadata"):
        meta = response.metadata
        vertex_count = meta.vertex_count
        face_count = meta.face_count
        generation_time_ms = meta.generation_time_ms
        intermediate = meta.intermediate_image_png if meta.intermediate_image_png else None
    else:
        vertex_count = 0
        face_count = 0
        generation_time_ms = 0.0
        intermediate = None

    return MeshResult(
        glb_data=response.glb_data,
        vertex_count=vertex_count,
        face_count=face_count,
        generation_time_ms=generation_time_ms,
        intermediate_image_png=intermediate,
    )
