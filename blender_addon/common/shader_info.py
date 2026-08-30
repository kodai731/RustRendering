"""Shared shader helper functions for GPUShaderCreateInfo construction.

Provides split_typedef_and_body, push_prelude, and specialize_body used by both
flame and water effect shaders to correctly construct Blender GPU shaders."""


def split_typedef_and_body(glsl_text: str) -> tuple[str, str]:
    """Split GLSL text into typedef (struct definitions) and body (the rest).

    Finds the last '};' closing a struct definition and splits there.
    Returns (typedef, body) where typedef ends with '};'."""
    lines = glsl_text.split("\n")
    last_struct_end = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "};":
            last_struct_end = i
    if last_struct_end < 0:
        raise ValueError("No closing '};' found in GLSL text")
    typedef = "\n".join(lines[: last_struct_end + 1])
    body = "\n".join(lines[last_struct_end + 1 :])
    return typedef, body


def push_prelude(struct_name: str, members: list[str]) -> str:
    """Return GLSL prelude declaring a const push constant struct.

    The values are placeholder zeros — specialize_body replaces references with
    actual specialization constants at compile time."""
    member_defs = "; ".join(members)
    zero_args = ", ".join("0" for _ in members)
    return (
        f"struct {struct_name} {{ {member_defs}; }};\n"
        f"const {struct_name} push = {struct_name}({zero_args});\n"
    )


def specialize_body(body: str, specialization: dict[str, float]) -> str:
    """Replace every reference to specialized uniforms with their compile-time values.

    Raises ValueError if a specialized uniform is not referenced by the shader body."""
    for uniform_path, value in specialization.items():
        if uniform_path not in body:
            raise ValueError(f"specialized uniform not referenced by the shader: {uniform_path}")
        body = body.replace(uniform_path, f"({float(value)!r})")
    return body
