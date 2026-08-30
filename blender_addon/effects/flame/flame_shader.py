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
        "const FlamePush push = FlamePush(0, 0, 0);\n"
    )


def specialize_body(body: str, specialization: dict[str, float]) -> str:
    for uniform_path, value in specialization.items():
        if uniform_path not in body:
            raise ValueError(f"specialized uniform not referenced by the shader: {uniform_path}")
        body = body.replace(uniform_path, f"({float(value)!r})")
    return body


def specialization_key(specialization: dict[str, float]) -> tuple:
    return tuple(sorted((name, float(value)) for name, value in specialization.items()))


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


def build_flame_shader(glsl_path: str, bindings_path: str, specialization: dict[str, float]):
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
    iface = gpu.types.GPUStageInterfaceInfo("flame_iface")
    iface.smooth("VEC2", "fragTexCoord")
    info.vertex_in(0, "VEC2", "pos")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "outColor")
    info.fragment_out(1, "VEC4", "outHistory")
    info.vertex_source("void main(){ fragTexCoord = pos*0.5+0.5; gl_Position = vec4(pos,0.0,1.0); }")
    info.fragment_source(push_prelude() + specialize_body(body, specialization))
    return gpu.shader.create_from_info(info)


def depth_convert_fragment_source() -> str:
    return (
        "void main(){"
        " float d = texture(windowDepth, fragTexCoord).r;"
        " float zEye = depthParams.y / (2.0 * d - 1.0 + depthParams.x);"
        " float engineDepth = (d >= 1.0 || zEye <= 0.0) ? 0.0 : depthParams.z / zEye;"
        " outDepth = vec4(engineDepth, 0.0, 0.0, 1.0);"
        " }"
    )


def build_depth_convert_shader():
    import gpu

    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, "FLOAT_2D", "windowDepth")
    info.push_constant("VEC3", "depthParams")
    iface = gpu.types.GPUStageInterfaceInfo("depth_convert_iface")
    iface.smooth("VEC2", "fragTexCoord")
    info.vertex_in(0, "VEC2", "pos")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "outDepth")
    info.vertex_source("void main(){ fragTexCoord = pos*0.5+0.5; gl_Position = vec4(pos,0.0,1.0); }")
    info.fragment_source(depth_convert_fragment_source())
    return gpu.shader.create_from_info(info)


def tonemap_composite_fragment_source() -> str:
    return (
        "vec3 acesFilmic(vec3 x){"
        " return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);"
        " }"
        "vec3 encodeSrgb(vec3 c){"
        " return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));"
        " }"
        "void main(){"
        " vec4 hdr = texture(image, fragTexCoord);"
        " vec3 display = acesFilmic(hdr.rgb * tonemapParams.x);"
        " if (tonemapParams.y > 0.5) { display = encodeSrgb(display); }"
        " outColor = vec4(display, hdr.a);"
        " }"
    )


def build_tonemap_composite_shader():
    import gpu

    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, "FLOAT_2D", "image")
    info.push_constant("MAT4", "ModelViewProjectionMatrix")
    info.push_constant("VEC2", "tonemapParams")
    iface = gpu.types.GPUStageInterfaceInfo("tonemap_composite_iface")
    iface.smooth("VEC2", "fragTexCoord")
    info.vertex_in(0, "VEC2", "pos")
    info.vertex_in(1, "VEC2", "texCoord")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "outColor")
    info.vertex_source(
        "void main(){ fragTexCoord = texCoord; gl_Position = ModelViewProjectionMatrix * vec4(pos, 0.0, 1.0); }"
    )
    info.fragment_source(tonemap_composite_fragment_source())
    return gpu.shader.create_from_info(info)
