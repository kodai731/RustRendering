import os
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

SHARED_DATA_ENV_VAR = "THYLLORE_SHARED_DATA_DIR"
SOURCE_MODEL_FILENAME = "curve_copilot_20260531_rawfuture_v1_tangent.onnx"
DUMMY_MODEL_FILENAME = "curve_copilot_20260531_dummy_rawfuture_v1_tangent.onnx"


def zero_all_initializers(model):
    for initializer in model.graph.initializer:
        original = numpy_helper.to_array(initializer)
        zeroed = np.zeros_like(original)
        initializer.CopyFrom(numpy_helper.from_array(zeroed, initializer.name))


def main():
    shared_data_dir = os.environ.get(SHARED_DATA_ENV_VAR)
    if not shared_data_dir:
        print(
            f"{SHARED_DATA_ENV_VAR} is not set. Set it (e.g. in your shell rc) to your "
            f"SharedData directory so that $DIR/exports/{SOURCE_MODEL_FILENAME} exists.",
            file=sys.stderr,
        )
        return 1

    exports_dir = Path(shared_data_dir) / "exports"
    source_model = exports_dir / SOURCE_MODEL_FILENAME
    dummy_model = exports_dir / DUMMY_MODEL_FILENAME

    if not source_model.exists():
        print(f"source model not found: {source_model}", file=sys.stderr)
        return 1

    model = onnx.load(str(source_model))
    zero_all_initializers(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(dummy_model))

    print(f"created {dummy_model} ({dummy_model.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
