"""Shared shader helper functions for GPUShaderCreateInfo construction.

Provides split_typedef_and_body, push_prelude, and specialize_body used by both
flame and water effect shaders to correctly construct Blender GPU shaders."""


def split_typedef_and_body(glsl_text: str) -> tuple[str, str]:
    """Split GLSL text into typedef (struct definitions) and body (the rest).

    Iterates through all lines: every top-level struct block (from `struct Name {`
    to its closing `};`) is collected into the typedef string in order of appearance.
    All other lines go into the body string in their original order.
    Raises ValueError if no struct definitions are found."""
    lines = glsl_text.split("\n")
    typedef_lines: list[str] = []
    body_lines: list[str] = []
    in_struct = False
    for line in lines:
        stripped = line.strip()
        if not in_struct and stripped.startswith("struct"):
            in_struct = True
            typedef_lines.append(line)
        elif in_struct:
            typedef_lines.append(line)
            if "}" in stripped:
                in_struct = False
        else:
            body_lines.append(line)
    if not typedef_lines:
        raise ValueError("No struct definitions found in GLSL text")
    typedef = "\n".join(typedef_lines)
    body = "\n".join(body_lines)
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
