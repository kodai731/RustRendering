"""
Blender headless smoke test invoked from tests/blender_grpc_smoke_tests.rs:

    blender --background --python blender_addon/tests/smoke_auto_rig.py -- <port> <output_glb_path>

Environment:
    PYTHONPATH must contain the repo root so `import blender_addon.grpc_client` works.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def parse_args() -> tuple[int, Path]:
    argv = sys.argv
    idx = argv.index("--") if "--" in argv else len(argv)
    args = argv[idx + 1:]
    if len(args) < 2:
        print("Usage: ... -- <port> <output_glb_path>", file=sys.stderr)
        sys.exit(2)
    return int(args[0]), Path(args[1])


def export_default_cube_to_glb_bytes() -> bytes:
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add()

    tmp_path = Path(bpy.app.tempdir) / "_smoke_cube.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(tmp_path),
        export_format="GLB",
        use_selection=False,
    )
    return tmp_path.read_bytes()


def main() -> int:
    port, output_path = parse_args()

    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        for p in pythonpath.split(os.pathsep):
            if p and p not in sys.path:
                sys.path.insert(0, p)

    try:
        from blender_addon.grpc_client import (
            AutoRigClient,
            AutoRigInput,
            ServerConfig,
        )
    except ImportError:
        print("[smoke] cannot import blender_addon.grpc_client", file=sys.stderr)
        traceback.print_exc()
        return 1

    try:
        glb_bytes = export_default_cube_to_glb_bytes()
        print(f"[smoke] exported cube GLB ({len(glb_bytes)} bytes)")

        config = ServerConfig(host="127.0.0.1", port=port, deadline_seconds=10.0)
        client = AutoRigClient(config)
        result = client.generate_rig(AutoRigInput(
            glb_data=glb_bytes,
            num_sample_points=1024,
        ))
        print(f"[smoke] received: bones={result.bone_count}, joints={result.joint_count}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(result.rigged_glb_data)
        print(f"[smoke] wrote {len(result.rigged_glb_data)} bytes to {output_path}")
        return 0

    except Exception:
        traceback.print_exc()
        return 1


sys.exit(main())
