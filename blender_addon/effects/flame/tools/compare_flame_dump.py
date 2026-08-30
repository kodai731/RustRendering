import argparse
import json
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare engine vs Blender flame dump")
    parser.add_argument("--engine", required=True, help="Path to engine .npy file")
    parser.add_argument("--blender", required=True, help="Path to Blender .npy file")
    parser.add_argument("--out", required=True, help="Output directory for PGM")
    parser.add_argument("--flip-blender", action="store_true", help="Flip Blender output vertically")
    args = parser.parse_args()

    engine = np.load(args.engine)
    blender = np.load(args.blender)

    if args.flip_blender:
        blender = blender[::-1]

    if engine.shape != blender.shape:
        print(f"Shape mismatch: engine {engine.shape} vs blender {blender.shape}", file=sys.stderr)
        sys.exit(1)

    h, w, c = engine.shape
    assert c == 4, f"Expected 4 channels, got {c}"

    diff = np.abs(engine.astype(np.float64) - blender.astype(np.float64))

    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    engine_alpha_count = int(np.sum(engine[:, :, 3] > 0))
    blender_alpha_count = int(np.sum(blender[:, :, 3] > 0))

    def alpha_bbox(img):
        rows, cols = np.where(img[:, :, 3] > 0)
        if len(rows) == 0:
            return None
        return {
            "y_min": int(np.min(rows)),
            "x_min": int(np.min(cols)),
            "y_max": int(np.max(rows)),
            "x_max": int(np.max(cols)),
        }

    engine_bbox = alpha_bbox(engine)
    blender_bbox = alpha_bbox(blender)

    summary = {
        "shape": list(engine.shape),
        "abs_diff_max": max_diff,
        "abs_diff_mean": mean_diff,
        "engine_alpha_count": engine_alpha_count,
        "blender_alpha_count": blender_alpha_count,
        "engine_bbox": engine_bbox,
        "blender_bbox": blender_bbox,
    }

    print(json.dumps(summary, indent=2))

    os.makedirs(args.out, exist_ok=True)
    pgm_path = os.path.join(args.out, "diff_alpha.pgm")

    alpha_diff = diff[:, :, 3]
    alpha_diff_clipped = np.clip(alpha_diff, 0.0, 1.0)
    alpha_diff_8bit = (alpha_diff_clipped * 255.0).astype(np.uint8)

    with open(pgm_path, "wb") as f:
        f.write(b"P5\n")
        f.write(f"{w} {h}\n".encode("ascii"))
        f.write(b"255\n")
        f.write(alpha_diff_8bit.tobytes())

    print(f"PGM saved to {pgm_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
