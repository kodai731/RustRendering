"""Tests for flame_shader.py (bpy-free).

Verifies split_typedef_and_body against the exporter output and pack_frame_ubo
byte layout."""

import os
import subprocess
import sys
import tempfile

import pytest

from blender_addon.common.shader_info import (
    push_prelude,
    specialize_body,
    split_typedef_and_body,
)
from blender_addon.effects.flame.flame_shader import (
    pack_frame_ubo,
    specialization_key,
)


SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "blender", "flame", "export_glsl.py")


def _run_exporter(tmp_path: str) -> str:
    """Run the exporter into tmp_path and return the GLSL text."""
    out_dir = os.path.join(tmp_path, "shaders")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--repo-root", os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."), "--out", out_dir],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"exporter failed: {result.stderr}"
    with open(os.path.join(out_dir, "flame_resolve.glsl")) as f:
        return f.read()


def test_split_typedef_ends_with_semicolon():
    """typedef part must end with '};'."""
    with tempfile.TemporaryDirectory() as tmp:
        glsl_text = _run_exporter(tmp)
    typedef, body = split_typedef_and_body(glsl_text)
    assert typedef.strip().endswith("};"), "typedef must end with '};"


def test_split_typedef_contains_flame_ubo():
    """typedef part must contain 'struct FlameUBO'."""
    with tempfile.TemporaryDirectory() as tmp:
        glsl_text = _run_exporter(tmp)
    typedef, body = split_typedef_and_body(glsl_text)
    assert "struct FlameUBO" in typedef, "typedef must contain struct FlameUBO"


def test_split_body_does_not_contain_flame_ubo():
    """body part must not contain 'struct FlameUBO'."""
    with tempfile.TemporaryDirectory() as tmp:
        glsl_text = _run_exporter(tmp)
    typedef, body = split_typedef_and_body(glsl_text)
    assert "struct FlameUBO" not in body, "body must not contain struct FlameUBO"


def test_pack_frame_ubo_size():
    """pack_frame_ubo must return exactly 176 bytes."""
    identity = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    data = pack_frame_ubo(identity, identity, (0.0, 0.0, -5.0, 1.0), (1.0, 1.0, 1.0, 0.0), (1.0, 1.0, 1.0, 1.0))
    assert len(data) == 176, f"expected 176 bytes, got {len(data)}"


def test_pack_frame_ubo_column_major():
    """view[0][1] (row 0, col 1) must appear at float index 4 in column-major layout.

    In row-major: view[0][1] is the 2nd float (index 1).
    In column-major: view[0][1] is the 5th float (index 4) — it's the 2nd element of column 0."""
    view = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]
    identity = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    data = pack_frame_ubo(view, identity, (0.0, 0.0, -5.0, 1.0), (1.0, 1.0, 1.0, 0.0), (1.0, 1.0, 1.0, 1.0))
    import struct
    values = struct.unpack("44f", data)
    # view[0][1] = 2.0 should be at index 4 in column-major
    assert values[4] == 2.0, f"view[0][1]=2.0 expected at index 4, got {values[4]}"


def test_push_prelude_fixes_mode_zero_at_compile_time():
    prelude = push_prelude("FlamePush", ["int mode", "int stepCount", "int debugView"])
    assert "const FlamePush push = FlamePush(0, 0, 0);" in prelude
    assert "#define" not in prelude


def test_specialize_body_replaces_every_reference():
    body = "if (flame.contourParams.rteBands >= 2.0) x = flame.contourParams.rteBands;"
    out = specialize_body(body, {"flame.contourParams.rteBands": 4.0})
    assert "flame.contourParams.rteBands" not in out
    assert out.count("(4.0)") == 2


def test_specialize_body_rejects_unreferenced_uniform():
    with pytest.raises(ValueError):
        specialize_body("void main() {}", {"flame.emitterParams.kind": 0.0})


def test_specialization_key_is_order_independent():
    a = specialization_key({"b": 1.0, "a": 0.0})
    b = specialization_key({"a": 0, "b": 1})
    assert a == b == (("a", 0.0), ("b", 1.0))


def test_exported_body_references_all_specialized_uniforms(tmp_path):
    _, body = split_typedef_and_body(_run_exporter(str(tmp_path)))
    for name in ("flame.contourParams.rteBands", "flame.trailMeta.sampleCount", "flame.emitterParams.kind"):
        assert name in body
