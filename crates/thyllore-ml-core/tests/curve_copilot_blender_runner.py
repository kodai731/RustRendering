import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def parse_argv_after_double_dash():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", required=True)
    parser.add_argument("--fixture-root", required=True)
    parser.add_argument("--case-id", required=True, choices=["short", "medium", "long"])
    parser.add_argument("--typed-output", required=True)
    parser.add_argument("--call-op-output", required=True)
    return parser.parse_args(argv)


def find_thyllore_ml_core_wheel(wheel_dir):
    wheel_dir = Path(wheel_dir)
    candidates = sorted(wheel_dir.glob("thyllore_ml_core-*.whl"))
    if not candidates:
        raise RuntimeError(f"no thyllore_ml_core wheel under {wheel_dir}")
    if sys.platform == "linux":
        for path in candidates:
            if "manylinux" in path.name or "linux" in path.name:
                return path
    elif sys.platform == "win32":
        for path in candidates:
            if "win_amd64" in path.name:
                return path
    elif sys.platform == "darwin":
        for path in candidates:
            if "macosx" in path.name:
                return path
    return candidates[-1]


def insert_wheel_into_sys_path(wheel_path):
    extract_dir = Path(tempfile.mkdtemp(prefix="thyllore_ml_core_phase5_"))
    with zipfile.ZipFile(wheel_path) as zf:
        zf.extractall(extract_dir)
    sys.path.insert(0, str(extract_dir))
    return extract_dir


def stage_onnxruntime_dylib_next_to_pyd(extract_dir):
    dylib_path = os.environ.get("ORT_DYLIB_PATH")
    if not dylib_path or not Path(dylib_path).exists():
        return
    pyd_dir = extract_dir / "thyllore_ml_core"
    if not pyd_dir.exists():
        return

    source_dir = Path(dylib_path).parent
    for sibling in source_dir.iterdir():
        if not sibling.is_file():
            continue
        if sibling.suffix.lower() not in (".dll", ".so", ".dylib"):
            continue
        target = pyd_dir / sibling.name
        if not target.exists():
            shutil.copy2(sibling, target)

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(pyd_dir))
    os.environ["ORT_DYLIB_PATH"] = str(pyd_dir / Path(dylib_path).name)


def read_input_fixture(fixture_root, case_id):
    path = Path(fixture_root) / f"numpy/curve_copilot_input_{case_id}.json"
    return json.loads(path.read_text())


def resolve_model_path(fixture_root, input_fixture):
    relative = input_fixture["onnx_relative_path"]
    return str((Path(fixture_root) / relative).resolve()).replace("\\", "/")


def u32_bits_to_float32_array(bits_list):
    import numpy as np
    return np.array(bits_list, dtype=np.uint32).view(np.float32)


def float32_array_to_u32_bits_list(array):
    import numpy as np
    return np.asarray(array, dtype=np.float32).view(np.uint32).tolist()


def build_typed_predictions_json(predictions_2d, confidences_1d):
    predictions = []
    for row, conf in zip(predictions_2d, confidences_1d):
        bits = float32_array_to_u32_bits_list(row)
        predictions.append({
            "value_bits": bits[0],
            "tangent_in_bits": [bits[1], bits[2]],
            "tangent_out_bits": [bits[3], bits[4]],
            "confidence_bits": float32_array_to_u32_bits_list([conf])[0],
        })
    return {"predictions": predictions}


def run_typed_path(tml, input_fixture, model_path):
    context = u32_bits_to_float32_array(input_fixture["context_bits"])
    topology = u32_bits_to_float32_array(input_fixture["topology_features_bits"])
    query_times = u32_bits_to_float32_array(input_fixture["query_times_bits"])
    curve_window = u32_bits_to_float32_array(input_fixture["curve_window_bits"])

    import numpy as np
    bone_tokens = np.array(input_fixture["bone_name_tokens"], dtype=np.int64)

    session = tml.PySession.from_onnx_path(model_path)
    predictions_2d, confidences_1d = session.run_curve_copilot(
        context,
        input_fixture["property_type"],
        topology,
        bone_tokens,
        query_times,
        curve_window,
    )
    return build_typed_predictions_json(predictions_2d, confidences_1d)


def run_call_op_json_path(tml, input_fixture, model_path):
    payload = {
        "model_path": model_path,
        "property_type": input_fixture["property_type"],
        "context_bits": input_fixture["context_bits"],
        "topology_features_bits": input_fixture["topology_features_bits"],
        "bone_name_tokens": input_fixture["bone_name_tokens"],
        "query_times_bits": input_fixture["query_times_bits"],
        "curve_window_bits": input_fixture["curve_window_bits"],
    }
    return tml.call_op_json("run_curve_copilot", payload)


def write_predictions_json(path, predictions_dict):
    Path(path).write_bytes((json.dumps(predictions_dict, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def main():
    args = parse_argv_after_double_dash()

    wheel_path = find_thyllore_ml_core_wheel(args.wheel_dir)
    extract_dir = insert_wheel_into_sys_path(wheel_path)
    stage_onnxruntime_dylib_next_to_pyd(extract_dir)

    import thyllore_ml_core as tml

    abi_marker = tml.__abi_marker__
    if abi_marker != 1:
        print(
            f"ERROR: __abi_marker__ mismatch (got {abi_marker}, expected 1)",
            file=sys.stderr,
        )
        sys.exit(2)

    capabilities = tml.capabilities()
    if "curve_copilot" not in capabilities:
        print(
            f"ERROR: capabilities() missing curve_copilot: {capabilities}",
            file=sys.stderr,
        )
        sys.exit(3)

    input_fixture = read_input_fixture(args.fixture_root, args.case_id)
    model_path = resolve_model_path(args.fixture_root, input_fixture)

    typed_predictions = run_typed_path(tml, input_fixture, model_path)
    write_predictions_json(args.typed_output, typed_predictions)

    call_op_predictions = run_call_op_json_path(tml, input_fixture, model_path)
    write_predictions_json(args.call_op_output, call_op_predictions)

    print(f"OK: case={args.case_id} typed={args.typed_output} call_op={args.call_op_output}")


if __name__ == "__main__":
    main()
