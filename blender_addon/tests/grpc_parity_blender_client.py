"""gRPC parity script - runs inside Blender 4.2 LTS embedded Python.

Sends the same fixture-derived proto requests that ``grpc_parity_rust_client.rs``
sends, then writes a canonical JSON of the response so the orchestrator can
compare both files byte-for-byte.

Invoked by the orchestrator (``grpc_request_wire_parity.rs``):

    blender --background --factory-startup \\
        --python grpc_parity_blender_client.py -- \\
        --server-url http://127.0.0.1:50051 \\
        --fixture-root <SharedDataPath>/fixtures/ml_parity \\
        --result-dir <temp dir>

The script imports proto stubs directly (not the dataclass adapters in
``thyllore_animation.grpc_client.{auto_rig,mesh,motion}``) so the request wire
bytes are bit-identical to the Rust tonic client's encoding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

RIGGING_REQUEST_FIXTURE = "proto/rigging_request.bin"
MOTION_REQUEST_FIXTURE = "proto/motion_request.bin"
MESH_REQUEST_FIXTURE = "proto/mesh_request.bin"

RIGGING_RESULT_NAME = "rigging_response_blender.json"
MOTION_RESULT_NAME = "motion_response_blender.json"
MESH_RESULT_NAME = "mesh_response_blender.json"


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--fixture-root", required=True)
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args(argv)


def f32_to_u32(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def sha256_hex(data):
    return hashlib.sha256(data or b"").hexdigest()


def write_canonical_json(path, value):
    text = json.dumps(value, sort_keys=True, indent=2) + "\n"
    Path(path).write_text(text)


def enable_addon_and_import_stubs():
    """Enable the thyllore_animation add-on (legacy or extension) and return its proto stubs.

    The extension install path differs between Blender 4.0 (legacy add-on,
    package = ``thyllore_animation``) and 4.2+ (extension, package =
    ``bl_ext.user_default.thyllore_animation``). Try both and import stubs
    from whichever loads.
    """
    import bpy
    import addon_utils
    import importlib

    candidates = [
        ("bl_ext.user_default.thyllore_animation", "bl_ext.user_default.thyllore_animation.grpc_client.stubs"),
        ("thyllore_animation", "thyllore_animation.grpc_client.stubs"),
    ]
    last_error = None
    for module_name, stubs_module in candidates:
        loaded_default, loaded_state = addon_utils.check(module_name)
        if not (loaded_default or loaded_state):
            try:
                bpy.ops.preferences.addon_enable(module=module_name)
            except RuntimeError as e:
                last_error = e
                continue
        try:
            stubs_pkg = importlib.import_module(stubs_module)
            pb2 = importlib.import_module(stubs_module + ".animation_ml_pb2")
            pb2_grpc = importlib.import_module(stubs_module + ".animation_ml_pb2_grpc")
            return pb2, pb2_grpc
        except ImportError as e:
            last_error = e
            continue
    raise RuntimeError(
        f"thyllore_animation add-on / proto stubs not importable; last error: {last_error}"
    )


def open_channel(server_url):
    import grpc  # type: ignore

    target = server_url
    for prefix in ("http://", "https://"):
        if target.startswith(prefix):
            target = target[len(prefix):]
            break
    return grpc.insecure_channel(target)


def auto_rig_response_to_canonical(response):
    return {
        "rigged_glb_sha256": sha256_hex(response.rigged_glb_data),
        "rigged_glb_size": len(response.rigged_glb_data),
        "metadata": rigging_metadata_to_canonical(response.metadata),
        "skeleton_joints": [
            {
                "name": j.name,
                "parent_index": j.parent_index,
                "head_x_bits": f32_to_u32(j.x),
                "head_y_bits": f32_to_u32(j.y),
                "head_z_bits": f32_to_u32(j.z),
                "tail_x_bits": f32_to_u32(j.tail_x),
                "tail_y_bits": f32_to_u32(j.tail_y),
                "tail_z_bits": f32_to_u32(j.tail_z),
            }
            for j in response.skeleton_joints
        ],
    }


def rigging_metadata_to_canonical(metadata):
    if metadata is None or not metadata.ByteSize():
        return None
    return {
        "joint_count": metadata.joint_count,
        "bone_count": metadata.bone_count,
        "generation_time_ms_bits": f32_to_u32(metadata.generation_time_ms),
    }


def motion_response_to_canonical(response):
    return {
        "model_used": response.model_used,
        "generation_time_ms_bits": f32_to_u32(response.generation_time_ms),
        "curves": [
            {
                "bone_name": curve.bone_name,
                "property_type": int(curve.property_type),
                "keyframes": [
                    {
                        "time_bits": f32_to_u32(kf.time),
                        "value_bits": f32_to_u32(kf.value),
                        "tangent_in_dt_bits": f32_to_u32(kf.tangent_in_dt),
                        "tangent_in_dv_bits": f32_to_u32(kf.tangent_in_dv),
                        "tangent_out_dt_bits": f32_to_u32(kf.tangent_out_dt),
                        "tangent_out_dv_bits": f32_to_u32(kf.tangent_out_dv),
                        "interpolation": int(kf.interpolation),
                    }
                    for kf in curve.keyframes
                ],
            }
            for curve in response.curves
        ],
    }


def mesh_response_to_canonical(response):
    return {
        "glb_sha256": sha256_hex(response.glb_data),
        "glb_size": len(response.glb_data),
        "metadata": mesh_metadata_to_canonical(response.metadata),
    }


def mesh_metadata_to_canonical(metadata):
    if metadata is None or not metadata.ByteSize():
        return None
    return {
        "vertex_count": metadata.vertex_count,
        "face_count": metadata.face_count,
        "generation_time_ms_bits": f32_to_u32(metadata.generation_time_ms),
        "intermediate_image_png_sha256": sha256_hex(metadata.intermediate_image_png),
        "intermediate_image_png_size": len(metadata.intermediate_image_png or b""),
    }


def load_proto_request(message_class, fixture_path):
    request = message_class()
    request.ParseFromString(Path(fixture_path).read_bytes())
    return request


def run_auto_rig(stubs, channel, fixture_root, result_dir):
    pb2, pb2_grpc = stubs
    request = load_proto_request(
        pb2.RiggingRequest,
        fixture_root / RIGGING_REQUEST_FIXTURE,
    )
    stub = pb2_grpc.AutoRiggingServiceStub(channel)
    response = stub.GenerateRig(request)
    write_canonical_json(
        result_dir / RIGGING_RESULT_NAME,
        auto_rig_response_to_canonical(response),
    )


def run_text_to_motion(stubs, channel, fixture_root, result_dir):
    pb2, pb2_grpc = stubs
    request = load_proto_request(
        pb2.MotionRequest,
        fixture_root / MOTION_REQUEST_FIXTURE,
    )
    stub = pb2_grpc.TextToMotionServiceStub(channel)
    response = stub.GenerateMotion(request)
    write_canonical_json(
        result_dir / MOTION_RESULT_NAME,
        motion_response_to_canonical(response),
    )


def run_mesh(stubs, channel, fixture_root, result_dir):
    pb2, pb2_grpc = stubs
    request = load_proto_request(
        pb2.MeshRequest,
        fixture_root / MESH_REQUEST_FIXTURE,
    )
    stub = pb2_grpc.MeshGenerationServiceStub(channel)
    response = stub.GenerateMesh(request)
    write_canonical_json(
        result_dir / MESH_RESULT_NAME,
        mesh_response_to_canonical(response),
    )


def main():
    args = parse_args()
    fixture_root = Path(args.fixture_root)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    stubs = enable_addon_and_import_stubs()
    channel = open_channel(args.server_url)
    try:
        run_auto_rig(stubs, channel, fixture_root, result_dir)
        run_text_to_motion(stubs, channel, fixture_root, result_dir)
        run_mesh(stubs, channel, fixture_root, result_dir)
    finally:
        channel.close()

    print(f"OK: Tier A Blender client wrote results to {result_dir}")


if __name__ == "__main__":
    main()
