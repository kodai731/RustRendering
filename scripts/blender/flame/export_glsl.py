"""Export GLSL shaders for the flame effect into Blender-compatible form."""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from blender_addon.common.glsl_export import expand_includes, strip_include_guards  # noqa: E402


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
                r'^\s*layout\s*\(\s*set\s*=\s*\d+\s*,\s*binding\s*=\s*\d+\s*\)\s+uniform\s+(\w+)\s*\{',
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GLSL shaders for Blender addon")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--out", required=True, help="Output directory for generated files")
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    out_dir = os.path.abspath(args.out)

    expanded_lines = expand_includes("flame/flameResolveFragment.frag", repo_root)

    stripped_lines = strip_include_guards(expanded_lines)

    output_lines, bindings = convert_to_blender_dialect(stripped_lines)

    os.makedirs(out_dir, exist_ok=True)
    glsl_path = os.path.join(out_dir, "flame_resolve.glsl")
    with open(glsl_path, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

    json_path = os.path.join(out_dir, "flame_resolve.bindings.json")
    with open(json_path, "w") as f:
        json.dump(bindings, f, indent=2)
        f.write("\n")

    print(f"Written {len(output_lines)} lines to {glsl_path}")
    print(f"Bindings written to {json_path}")


if __name__ == "__main__":
    main()
