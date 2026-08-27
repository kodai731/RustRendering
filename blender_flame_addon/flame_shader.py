import json
import struct


def split_typedef_and_body(glsl_text: str) -> tuple[str, str]:
    lines = glsl_text.split("\n")
    flame_ubo_start = -1
    flame_ubo_end = -1
    for i, line in enumerate(lines):
        if "struct FlameUBO {" in line:
            flame_ubo_start = i
            break
    if flame_ubo_start < 0:
        raise ValueError("No 'struct FlameUBO {' found in GLSL text")
    for j in range(flame_ubo_start, len(lines)):
        stripped = lines[j].strip()
        if stripped == "};":
            flame_ubo_end = j
            break
    if flame_ubo_end < 0:
        raise ValueError("No closing '};' found after FlameUBO")
    typedef = "\n".join(lines[: flame_ubo_end + 1])
    body = "\n".join(lines[flame_ubo_end + 1 :])
    return typedef, body


def push_prelude() -> str:
    return (
        "struct FlamePush { int mode; int stepCount; int debugView; };\n"
        "#define push FlamePush(push_mode, push_stepCount, push_debugView)\n"
    )


def pack_frame_ubo(
    view: list[list[float]],
    proj: list[list[float]],
    camera_pos: tuple[float, float, float, float],
    light_pos: tuple[float, float, float, float],
    light_color: tuple[float, float, float, float],
) -> bytes:
    view_col = []
    for col in range(4):
        for row in range(4):
            view_col.append(view[row][col])
    proj_col = []
    for col in range(4):
        for row in range(4):
            proj_col.append(proj[row][col])
    values = view_col + proj_col + list(camera_pos) + list(light_pos) + list(light_color)
    assert len(values) == 16 * 2 + 4 * 3, f"expected 44 floats, got {len(values)}"
    return struct.pack("44f", *values)


def build_flame_shader(glsl_path: str, bindings_path: str):
    import bpy
    import gpu

    with open(glsl_path) as f:
        glsl_text = f.read()
    with open(bindings_path) as f:
        bindings = json.load(f)

    typedef, body = split_typedef_and_body(glsl_text)

    info = gpu.types.GPUShaderCreateInfo()
    info.typedef_source(typedef)
    info.uniform_buf(0, "FrameUBO", "frame")
    info.uniform_buf(1, "FlameUBO", "flame")
    for i, sampler in enumerate(bindings["samplers"]):
        info.sampler(i, "FLOAT_2D", sampler["name"])
    info.push_constant("INT", "push_mode")
    info.push_constant("INT", "push_stepCount")
    info.push_constant("INT", "push_debugView")
    iface = gpu.types.GPUStageInterfaceInfo("flame_iface")
    iface.smooth("VEC2", "fragTexCoord")
    info.vertex_in(0, "VEC2", "pos")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "outColor")
    info.fragment_out(1, "VEC4", "outHistory")
    info.vertex_source("void main(){ fragTexCoord = pos*0.5+0.5; gl_Position = vec4(pos,0.0,1.0); }")
    info.fragment_source(push_prelude() + body)
    return gpu.shader.create_from_info(info)
