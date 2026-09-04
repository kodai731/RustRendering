import argparse
import json
import os
import re
import sys


def resolve_layout_macros(line: str, defines: dict[str, str]) -> str:
    if not re.match(r'^\s*layout\s*\(', line):
        return line
    return re.sub(r'\b[A-Za-z_]\w*\b', lambda m: defines.get(m.group(0), m.group(0)), line)


def expand_includes(source_path: str, repo_root: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    defines: dict[str, str] = {}

    def _expand(path: str, text: str) -> None:
        if path in seen:
            return
        seen.add(path)
        for line in text.split("\n"):
            m_define = re.match(r'^\s*#\s*define\s+(\w+)\s+(\d+)\s*$', line)
            if m_define:
                defines[m_define.group(1)] = m_define.group(2)

            m = re.match(r'^\s*#\s*include\s+"([^"]+)"', line)
            if m:
                included = m.group(1)
                # Skip water_secondary.glsl — it contains ray-query code (traceScene, VertexBuffer)
                if included == "include/water_secondary.glsl":
                    continue
                base_dir = os.path.dirname(path)
                inc_path = os.path.normpath(os.path.join(base_dir, included))
                inc_full = os.path.join(repo_root, "shaders", inc_path)
                with open(inc_full, "r") as f:
                    _expand(inc_path, f.read())
            else:
                result.append(resolve_layout_macros(line, defines))

    entry = "waterResolveFragment.frag"
    full = os.path.join(repo_root, "shaders", entry)
    with open(full, "r") as f:
        _expand(entry, f.read())
    return result


def strip_include_guards(lines: list[str]) -> list[str]:
    """Strip #ifndef/#define _GLSL guards and #ifdef WATER_RAY_QUERY blocks.

    Returns (output_lines, has_endif) where has_endif is True if any #endif was consumed
    by a stripped block (used to detect unbalanced guards)."""
    result: list[str] = []
    stack: list[str] = []  # "guard" for _GLSL guards, "water_ray_query" for WATER_RAY_QUERY blocks
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        m_ifndef = re.match(r'^#\s*ifndef\s+(\S+)', stripped)
        if m_ifndef:
            macro = m_ifndef.group(1)
            is_guard = macro.endswith("_GLSL")
            stack.append("guard" if is_guard else "other")
            i += 1
            if not is_guard:
                result.append(line)
            if is_guard and i < len(lines):
                next_stripped = lines[i].strip()
                m_define = re.match(r'^#\s*define\s+' + re.escape(macro), next_stripped)
                if m_define:
                    i += 1
            continue

        m_ifdef = re.match(r'^#\s*ifdef\s+(\S+)', stripped)
        if m_ifdef:
            macro = m_ifdef.group(1)
            if macro == "WATER_RAY_QUERY":
                # Skip this block entirely — WATER_RAY_QUERY is not defined for Blender
                stack.append("water_ray_query")
                i += 1
                continue
            else:
                stack.append("other")
                result.append(line)
                i += 1
                continue

        m_if = re.match(r'^#\s*if\b', stripped)
        if m_if:
            stack.append("other")
            result.append(line)
            i += 1
            continue

        m_endif = re.match(r'^#\s*endif\b', stripped)
        if m_endif:
            if stack and stack[-1] == "water_ray_query":
                # Consume this endif — it closes a WATER_RAY_QUERY block we're skipping
                stack.pop()
                i += 1
                continue
            elif stack and stack[-1] == "guard":
                stack.pop()
                i += 1
                continue
            elif stack:
                stack.pop()
            result.append(line)
            i += 1
            continue

        # Skip lines inside a water_ray_query block
        if stack and stack[-1] == "water_ray_query":
            i += 1
            continue

        result.append(line)
        i += 1
    return result


def convert_to_blender_dialect(lines: list[str]) -> tuple[list[str], dict]:
    output: list[str] = []
    bindings: dict = {
        "samplers": [],
        "ubos": [],
        "push_constants": [],
        "inputs": [],
        "outputs": [],
    }

    i = 0
    while i < len(lines):
        line = lines[i]

        if re.match(r'^\s*#\s*version\b', line):
            i += 1
            continue

        layout_match = re.match(r'^\s*layout\s*\(', line)
        if layout_match:
            loc_in_match = re.match(
                r'^\s*layout\s*\(\s*location\s*=\s*(\d+)\s*\)\s+in\s+\w+\s+(\w+)\s*;', line
            )
            if loc_in_match:
                name = loc_in_match.group(2)
                bindings["inputs"].append(name)
                i += 1
                continue

            loc_out_match = re.match(
                r'^\s*layout\s*\(\s*location\s*=\s*(\d+)\s*\)\s+out\s+\w+\s+(\w+)\s*;', line
            )
            if loc_out_match:
                name = loc_out_match.group(2)
                bindings["outputs"].append(name)
                i += 1
                continue

            sampler_match = re.match(
                r'^\s*layout\s*\(\s*set\s*=\s*\d+\s*,\s*binding\s*=\s*(\d+)\s*\)\s+uniform\s+sampler2D\s+(\w+)\s*;',
                line,
            )
            if sampler_match:
                binding = int(sampler_match.group(1))
                name = sampler_match.group(2)
                bindings["samplers"].append({"name": name, "binding": binding})
                i += 1
                continue

            ubo_match = re.match(
                r'^\s*layout\s*\(\s*set\s*=\s*\w+\s*,\s*binding\s*=\s*\w+\s*\)\s+uniform\s+(\w+)\s*\{',
                line,
            )
            if ubo_match:
                type_name = ubo_match.group(1)
                block_lines = [line]
                j = i + 1
                while j < len(lines):
                    block_lines.append(lines[j])
                    if re.match(r'^\s*\}\s*\w+\s*;', lines[j]):
                        break
                    j += 1
                closing = block_lines[-1]
                inst_match = re.match(r'^\s*\}\s*(\w+)\s*;', closing)
                inst_name = inst_match.group(1) if inst_match else ""
                new_block = []
                for k, bl in enumerate(block_lines):
                    if k == 0:
                        new_line = re.sub(
                            r'^\s*layout\s*\([^)]*\)\s+uniform\s+',
                            "struct ",
                            bl,
                        )
                        new_block.append(new_line)
                    elif k == len(block_lines) - 1:
                        new_block.append("};")
                    else:
                        new_block.append(bl)
                output.extend(new_block)
                bindings["ubos"].append({"type": type_name, "name": inst_name})
                i = j + 1
                continue

            push_match = re.match(
                r'^\s*layout\s*\(\s*push_constant\s*\)\s+uniform\s+(\w+)\s*\{',
                line,
            )
            if push_match:
                type_name = push_match.group(1)
                j = i + 1
                while j < len(lines):
                    if re.match(r'^\s*\}\s*\w+\s*;', lines[j]):
                        break
                    j += 1
                closing = lines[j]
                inst_match = re.match(r'^\s*\}\s*(\w+)\s*;', closing)
                inst_name = inst_match.group(1) if inst_match else ""
                members: list[str] = []
                for k in range(i + 1, j):
                    stripped = lines[k].strip()
                    if stripped and not stripped.startswith("//"):
                        members.append(stripped.rstrip(";"))
                bindings["push_constants"].append(
                    {"type": type_name, "name": inst_name, "members": members}
                )
                i = j + 1
                continue

            output.append(line)
            i += 1
            continue

        output.append(line)
        i += 1

    return output, bindings


SCENE_COLOR_SAMPLE_PATTERN = re.compile(r'texture\s*\(\s*sceneColorSampler\s*,\s*(\w+)\s*\)')


def remap_scene_color_to_capture_rect(lines: list[str]) -> list[str]:
    """Blender captures only the water's screen rect, so full-screen uvs are remapped through sceneColorRect."""
    remapped = []
    for line in lines:
        remapped.append(SCENE_COLOR_SAMPLE_PATTERN.sub(r'texture(sceneColorSampler, (\1 - sceneColorRect.xy) * sceneColorRect.zw)', line))
    leftover = [line for line in remapped if "sceneColorSampler" in line and "sceneColorRect" not in line]
    if leftover:
        raise SystemExit(f"sceneColorSampler reads that the rect remap does not cover: {leftover}")
    return remapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GLSL shaders for Blender addon")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--out", required=True, help="Output directory for generated files")
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    out_dir = os.path.abspath(args.out)

    expanded_lines = expand_includes("waterResolveFragment.frag", repo_root)

    stripped_lines = strip_include_guards(expanded_lines)

    output_lines, bindings = convert_to_blender_dialect(stripped_lines)
    output_lines = remap_scene_color_to_capture_rect(output_lines)

    os.makedirs(out_dir, exist_ok=True)
    glsl_path = os.path.join(out_dir, "water_torus.glsl")
    with open(glsl_path, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

    json_path = os.path.join(out_dir, "water_torus.bindings.json")
    with open(json_path, "w") as f:
        json.dump(bindings, f, indent=2)
        f.write("\n")

    print(f"Written {len(output_lines)} lines to {glsl_path}")
    print(f"Bindings written to {json_path}")


if __name__ == "__main__":
    main()
