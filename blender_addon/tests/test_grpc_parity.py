from __future__ import annotations

import json
from pathlib import Path

import pytest

from blender_addon.grpc_client.stubs import pb2

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_pair(name: str) -> tuple[bytes, dict]:
    bin_path = FIXTURES_DIR / f"{name}.bin"
    json_path = FIXTURES_DIR / f"{name}.json"
    if not bin_path.exists():
        pytest.skip(
            f"fixture missing: {bin_path}. "
            f"Run `cargo test -p thyllore-grpc-client --features text-to-motion "
            f"--test grpc_parity_fixtures generate_parity_fixtures -- --include-ignored` first."
        )
    return bin_path.read_bytes(), json.loads(json_path.read_text())


def _serialize(message) -> bytes:
    return message.SerializeToString(deterministic=True)


def test_rigging_request_bit_identical() -> None:
    expected_bytes, fields = _load_pair("rigging_request")

    request = pb2.RiggingRequest(
        glb_data=bytes.fromhex(fields["glb_data_hex"]),
        params=pb2.RiggingParams(num_sample_points=fields["num_sample_points"]),
        model_type=getattr(pb2, fields["model_type"]),
    )

    actual_bytes = _serialize(request)
    assert actual_bytes == expected_bytes, (
        f"proto bytes mismatch: rust={expected_bytes.hex()} python={actual_bytes.hex()}"
    )


def test_motion_request_bit_identical() -> None:
    expected_bytes, fields = _load_pair("motion_request")

    request = pb2.MotionRequest(
        prompt=fields["prompt"],
        duration_seconds=fields["duration_seconds"],
        target_fps=fields["target_fps"],
        skeleton_type=getattr(pb2, fields["skeleton_type"]),
        bone_mappings=[
            pb2.BoneMapping(
                source_joint_index=mapping["source_joint_index"],
                target_bone_name=mapping["target_bone_name"],
            )
            for mapping in fields["bone_mappings"]
        ],
        glb_skeleton=pb2.GlbSkeletonSpec(
            glb_data=bytes.fromhex(fields["glb_skeleton_glb_data_hex"]),
            skeleton_cache_id=fields["glb_skeleton_cache_id"],
        ),
        internal_use_only=fields["internal_use_only"],
    )

    actual_bytes = _serialize(request)
    assert actual_bytes == expected_bytes, (
        f"proto bytes mismatch: rust={expected_bytes.hex()} python={actual_bytes.hex()}"
    )


def test_mesh_request_bit_identical() -> None:
    expected_bytes, fields = _load_pair("mesh_request")

    request = pb2.MeshRequest(
        prompt=fields["prompt"],
        params=pb2.MeshGenerationParams(
            target_faces=fields["target_faces"],
            seed=fields["seed"],
            image_size=fields["image_size"],
            image_inference_steps=fields["image_inference_steps"],
        ),
        input_image_png=bytes.fromhex(fields["input_image_png_hex"]),
        input_mode=getattr(pb2, fields["input_mode"]),
        model_type=getattr(pb2, fields["model_type"]),
        t2i_model_type=getattr(pb2, fields["t2i_model_type"]),
    )

    actual_bytes = _serialize(request)
    assert actual_bytes == expected_bytes, (
        f"proto bytes mismatch: rust={expected_bytes.hex()} python={actual_bytes.hex()}"
    )
