"""Shared shader helper functions for GPUShaderCreateInfo construction.

Provides split_typedef_and_body, push_prelude, and specialize_body used by both
flame and water effect shaders to correctly construct Blender GPU shaders."""

import re


def split_typedef_and_body(glsl_text: str) -> tuple[str, str]:
    """Split GLSL text into typedef (struct definitions and top-level const declarations) and body (the rest).

    Every struct block, from its `struct Name {` line to the line closing it with
    `};`, goes to the typedef string in order of appearance. Top-level const
    declarations (lines starting with `const`) are also extracted into the typedef
    string, including multi-line declarations that span until a terminating `;`.
    Every other line goes to the body string in its original order. GLSL structs
    never nest, so a single in-struct flag is enough.

    Splitting on a line position instead would be wrong: the typedef is expanded
    before the uniform block instances are declared, so it must never capture a
    function referencing them, and the exported shaders interleave struct blocks
    with functions in both orders.
    Raises ValueError if no struct definition is found or if one is left unclosed."""
    struct_start = re.compile(r"^struct\b")
    typedef_lines: list[str] = []
    body_lines: list[str] = []
    in_struct = False
    in_const = False
    struct_count = 0

    def _code_without_comment(s: str) -> str:
        """Return the line without its trailing line comment and spaces."""
        return s.split("//")[0].rstrip()

    for line in glsl_text.split("\n"):
        stripped = line.strip()
        code = _code_without_comment(stripped)

        if in_struct:
            typedef_lines.append(line)
            in_struct = not code.endswith("};")
        elif in_const:
            typedef_lines.append(line)
            in_const = not code.endswith(";")
        elif struct_start.match(stripped):
            typedef_lines.append(line)
            struct_count += 1
            in_struct = not code.endswith("};")
        elif line.startswith("const"):
            typedef_lines.append(line)
            in_const = not code.endswith(";")
        else:
            body_lines.append(line)

    if in_struct:
        raise ValueError("Unclosed struct definition in GLSL text")
    if struct_count == 0:
        raise ValueError("No struct definitions found in GLSL text")

    return "\n".join(typedef_lines), "\n".join(body_lines)


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
