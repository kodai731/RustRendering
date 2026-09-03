"""Tests for split_typedef_and_body in blender_addon/common/shader_info.py (bpy-free).

Verifies that struct blocks are fully captured in typedefs, function definitions
are not mixed into typedefs, top-level const declarations (including multi-line)
are extracted, order is preserved, and ValueError is raised for invalid inputs."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

# Load shader_info module directly from file path to avoid package conflicts.
_SHADER_INFO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "common", "shader_info.py"
)
_spec = importlib.util.spec_from_file_location("shader_info", _SHADER_INFO_PATH)
_shader_info = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["shader_info"] = _shader_info
_spec.loader.exec_module(_shader_info)  # type: ignore[attr-defined]

split_typedef_and_body = _shader_info.split_typedef_and_body


def test_struct_blocks_captured_in_typedefs():
    """Multiple struct blocks with functions between them: all struct members stay
    in typedef, none leak into body."""
    glsl = """
struct Light {
    vec3 position;
    float intensity;
};

void computeLight(Light light) {
    float d = length(light.position);
}

struct Shadow {
    float depth;
    float bias;
};

void computeShadow(Shadow shadow) {
    float v = shadow.depth + shadow.bias;
}
"""
    typedef, body = split_typedef_and_body(glsl)

    # Both struct blocks fully in typedef
    assert "struct Light" in typedef
    assert "vec3 position;" in typedef
    assert "float intensity;" in typedef
    assert "struct Shadow" in typedef
    assert "float depth;" in typedef
    assert "float bias;" in typedef

    # No struct member lines in body
    assert "   vec3 position;" not in body
    assert "    float intensity;" not in body
    assert "    float depth;" not in body
    assert "    float bias;" not in body

    # Functions are in body
    assert "void computeLight" in body
    assert "void computeShadow" in body


def test_function_definitions_not_in_typedefs():
    """Function definitions (including lines referencing UBO instances) must not
    appear in the typedef part."""
    glsl = """
struct WaterUBO {
    mat4 view;
    mat4 proj;
};

void main() {
    vec4 clipPos = water.proj * water.view * vec4(pos, 1.0);
}
"""
    typedef, body = split_typedef_and_body(glsl)

    # Function definition in body, not typedef
    assert "void main()" in body
    assert "void main()" not in typedef

    # UBO instance reference in body, not typedef
    assert "water.proj" in body
    assert "water.proj" not in typedef


def test_multiline_const_captured_as_one_block():
    """Top-level const declarations are captured in typedef, including multi-line
    array initializers like `const vec3 K[3] = vec3[3](` ... `);`."""
    glsl = """
const int N = 4;

const vec3 K[3] = vec3[3](
    vec3(1.0),
    vec3(2.0),
    vec3(3.0)
);

struct Foo {
    float arr[N];
};

void bar() {
}
"""
    typedef, body = split_typedef_and_body(glsl)

    # Single-line const in typedef
    assert "const int N = 4;" in typedef

    # Multi-line const fully captured as one block in typedef
    assert "const vec3 K[3] = vec3[3](" in typedef
    assert "    vec3(1.0)," in typedef
    assert "    vec3(2.0)," in typedef
    assert "    vec3(3.0)" in typedef
    assert ");" in typedef

    # No const lines in body
    assert "const int N" not in body
    assert "const vec3 K" not in body


def test_order_preserved():
    """A const used as array length must appear before the struct that uses it
    in the typedef (order of appearance preserved)."""
    glsl = """
const int COUNT = 8;

struct Bar {
    float values[COUNT];
};

void foo() {
}
"""
    typedef, body = split_typedef_and_body(glsl)

    const_pos = typedef.index("const int COUNT")
    struct_pos = typedef.index("struct Bar")
    assert const_pos < struct_pos, "const must appear before struct in typedef"


def test_value_error_no_structs():
    """Input with no struct definitions must raise ValueError."""
    glsl = """
void main() {
}
"""
    with pytest.raises(ValueError, match="No struct"):
        split_typedef_and_body(glsl)


def test_value_error_unclosed_struct():
    """Input with an unclosed struct definition must raise ValueError."""
    glsl = """
struct Foo {
    float x;

void bar() {
}
"""
    with pytest.raises(ValueError, match="Unclosed"):
        split_typedef_and_body(glsl)
