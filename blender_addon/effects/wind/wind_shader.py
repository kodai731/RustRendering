import json
import struct

from ._common import _import_shared

shader_info = _import_shared("shader_info")


def matrix_column_major(m) -> list[float]:
    """Flatten a 4x4 matrix into a 16-element list in column-major order."""
    out: list[float] = []
    for col in range(4):
        for row in range(4):
            out.append(m[row][col])
    return out


def pack_frame_ubo(
    view: list[list[float]],
    proj: list[list[float]],
    camera_pos: tuple[float, float, float, float],
    light_pos: tuple[float, float, float, float],
    light_color: tuple[float, float, float, float],
) -> bytes:
    values = matrix_column_major(view) + matrix_column_major(proj) + list(camera_pos) + list(light_pos) + list(light_color)
    assert len(values) == 16 * 2 + 4 * 3, f"expected 44 floats, got {len(values)}"
    return struct.pack("44f", *values)


def specialization_key(specialization: dict) -> tuple:
    return (specialization.get("mode", 0), specialization.get("debugView", 0))


def build_wind_shader(glsl_path: str, bindings_path: str, specialization: dict):
    import bpy
    import gpu

    with open(glsl_path) as f:
        glsl_text = f.read()
    with open(bindings_path) as f:
        bindings = json.load(f)

    typedef, body = shader_info.split_typedef_and_body(glsl_text)

    info = gpu.types.GPUShaderCreateInfo()
    info.typedef_source(typedef)
    info.uniform_buf(0, "FrameUBO", "frame")
    info.uniform_buf(1, "WindUBO", "wind")
    for i, sampler in enumerate(bindings["samplers"]):
        info.sampler(i, "FLOAT_2D", sampler["name"])
    iface = gpu.types.GPUStageInterfaceInfo("wind_iface")
    iface.smooth("VEC2", "fragTexCoord")
    info.vertex_in(0, "VEC2", "pos")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "outColor")
    info.push_constant("INT", "mode")
    info.push_constant("INT", "stepCount")
    info.push_constant("INT", "debugView")
    info.vertex_source("void main(){ fragTexCoord = pos*0.5+0.5; gl_Position = vec4(pos,0.0,1.0); }")
    pc = bindings["push_constants"][0]
    wanted = {"push.mode": specialization.get("mode", 0), "push.debugView": specialization.get("debugView", 0)}
    info.fragment_source(shader_info.push_prelude(pc["type"], pc["members"]) + shader_info.specialize_body(body, {k: v for k, v in wanted.items() if k in body}))
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
        " if (hdr.a <= 0.0) discard;"
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
