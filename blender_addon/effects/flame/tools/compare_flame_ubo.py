import argparse
import json
import math
import os
import re
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def _unpack_wheel():
    wheel_dir = REPO_ROOT / "blender_addon" / "effects" / "flame" / "wheels"
    wheels = sorted(wheel_dir.glob("thyllore_effect_core-*.whl"))
    if not wheels:
        print("No thyllore_effect_core wheel found", file=sys.stderr)
        sys.exit(1)
    site_dir = REPO_ROOT / "log" / "blender_flame_probe" / "site"
    site_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheels[0]) as zf:
        for entry in zf.namelist():
            if entry.startswith("thyllore_effect_core"):
                zf.extract(entry, str(site_dir))
    sys.path.insert(0, str(site_dir))


def _read_generated_rs():
    path = REPO_ROOT / "crates" / "thyllore-effect-core" / "src" / "flame" / "gpu" / "components" / "generated.rs"
    return path.read_text()


def _parse_nested_struct_sizes(source):
    sizes = {}
    for m in re.finditer(r"struct\s+(\w+)\s*\{([^}]*)\}", source, re.DOTALL):
        name = m.group(1)
        fields = m.group(2)
        total = 0
        for fm in re.finditer(r"pub\s+\w+:\s*(.+?)[,\n]", fields):
            ftype = fm.group(1).strip().split(" = ")[0].strip()
            if ftype == "Matrix4<f32>":
                total += 64
            elif m2 := re.match(r"\[\[f32;\s*(\d+)\];\s*(\d+)\]", ftype):
                inner = int(m2.group(1))
                outer = int(m2.group(2))
                total += inner * 4 * outer
            elif m2 := re.match(r"\[(\w+);\s*(\d+)\]", ftype):
                inner_name = m2.group(1)
                count = int(m2.group(2))
                if inner_name in sizes:
                    total += sizes[inner_name] * count
                elif inner_name == "f32":
                    total += 4 * count
                else:
                    total += 16 * count
            elif ftype == "f32":
                total += 4
            elif ftype in sizes:
                total += sizes[ftype]
            else:
                total += 4
        sizes[name] = total
    return sizes


def _parse_struct_fields(source, struct_name):
    m = re.search(r"struct\s+" + re.escape(struct_name) + r"\s*\{([^}]*)\}", source, re.DOTALL)
    if not m:
        return []
    fields_text = m.group(1)
    nested_sizes = _parse_nested_struct_sizes(source)
    layout = []
    offset = 0
    for fm in re.finditer(r"pub\s+(\w+):\s*(.+?)[,\n]", fields_text):
        fname = fm.group(1)
        ftype = fm.group(2).strip().split(" = ")[0].strip()
        if ftype == "Matrix4<f32>":
            size = 64
        elif m2 := re.match(r"\[\[f32;\s*(\d+)\];\s*(\d+)\]", ftype):
            inner = int(m2.group(1))
            outer = int(m2.group(2))
            size = inner * 4 * outer
        elif m2 := re.match(r"\[(\w+);\s*(\d+)\]", ftype):
            inner_name = m2.group(1)
            count = int(m2.group(2))
            if inner_name in nested_sizes:
                size = nested_sizes[inner_name] * count
            elif inner_name == "f32":
                size = 4 * count
            else:
                size = 16 * count
        elif ftype in nested_sizes:
            size = nested_sizes[ftype]
        elif ftype == "f32":
            size = 4
        else:
            size = 4
        layout.append((fname, offset, size))
        offset += size
    return layout


def _parse_flame_ubo_layout(source):
    return _parse_struct_fields(source, "FlameUBO")


def _field_at_offset(layout, source, offset):
    for fname, start, size in layout:
        if start <= offset < start + size:
            rel = offset - start
            nested_sizes = _parse_nested_struct_sizes(source)
            ftype = _get_field_type(source, "FlameUBO", fname)
            if ftype in nested_sizes:
                nested_layout = _parse_struct_fields(source, ftype)
                nested_info = _field_at_offset(nested_layout, source, rel)
                return {"field": f"{fname}.{nested_info['field']}", "offset": offset, "field_start": start + nested_info["field_start"], "relative": nested_info["relative"]}
            elif ftype.endswith("; 2]") or ftype.endswith("; 32]"):
                m = re.match(r"\[(\w+);\s*(\d+)\]", ftype)
                if m:
                    inner_name = m.group(1)
                    count = int(m.group(2))
                    if inner_name in nested_sizes:
                        elem_size = nested_sizes[inner_name]
                        elem_idx = rel // elem_size
                        elem_rel = rel % elem_size
                        nested_layout = _parse_struct_fields(source, inner_name)
                        nested_info = _field_at_offset(nested_layout, source, elem_rel)
                        return {"field": f"{fname}[{elem_idx}].{nested_info['field']}", "offset": offset, "field_start": start + elem_idx * elem_size + nested_info["field_start"], "relative": nested_info["relative"]}
            return {"field": fname, "offset": offset, "field_start": start, "relative": rel}
    return {"field": "out_of_bounds", "offset": offset, "field_start": None, "relative": None}


def _get_field_type(source, struct_name, field_name):
    m = re.search(r"struct\s+" + re.escape(struct_name) + r"\s*\{([^}]*)\}", source, re.DOTALL)
    if not m:
        return ""
    fields_text = m.group(1)
    for fm in re.finditer(r"pub\s+(\w+):\s*(.+?)[,\n]", fields_text):
        if fm.group(1) == field_name:
            return fm.group(2).strip().split(" = ")[0].strip()
    return ""


def main():
    parser = argparse.ArgumentParser(description="Compare engine UBO bytes vs packed bytes")
    parser.add_argument("--engine", required=True, help="Path to engine .ubo.bin file")
    parser.add_argument("--params", required=True, help="Path to engine JSONL dump file")
    parser.add_argument("--time", type=float, default=None, help="Override time from JSONL")
    args = parser.parse_args()

    _unpack_wheel()
    import thyllore_effect_core as fx

    with open(args.params) as f:
        lines = [line.strip() for line in f if line.strip()]
    record = json.loads(lines[-1])

    time = args.time if args.time is not None else record["time"]
    position = list(record["position"])
    rotation = list(record["rotation"])
    ep = record.get("effect_params", record)
    flame_preset = fx.flame_preset_params("campfire")
    params = {k: v for k, v in ep.items() if k in flame_preset}
    light_position = ep.get("light_position_world")
    frame_index = int(record.get("frame_index", 0))

    packed_bytes = fx.pack_flame_ubo(params, time, position, rotation, light_position=light_position, frame_index=frame_index)
    engine_bytes = Path(args.engine).read_bytes()

    length_match = len(engine_bytes) == len(packed_bytes)
    match_count = sum(1 for a, b in zip(engine_bytes, packed_bytes) if a == b)

    first_diff = None
    for i, (a, b) in enumerate(zip(engine_bytes, packed_bytes)):
        if a != b:
            first_diff = i
            break

    generated_source = _read_generated_rs()
    layout = _parse_flame_ubo_layout(generated_source)

    field_info = None
    if first_diff is not None:
        field_info = _field_at_offset(layout, generated_source, first_diff)

    result = {
        "engine_length": len(engine_bytes),
        "packed_length": len(packed_bytes),
        "length_match": length_match,
        "match_count": match_count,
        "total_bytes": len(engine_bytes),
        "first_diff_offset": first_diff,
        "field_at_first_diff": field_info,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
