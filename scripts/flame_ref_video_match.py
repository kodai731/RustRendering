"""Verify pillar_ref_seq frames against the source YouTube video.

usage:
  python scripts/flame_ref_video_match.py --video /tmp/thyllore_flame_ref/pillar_src.mp4 \
      [--seq assets/textures/flames/pillar_ref_seq] [--allowed 0,2] [--json out.json]

For every sequence frame the whole video is scanned for the best
zero-mean NCC match (luminance, source crop, downsampled). Reports the
matched video frame index / time, the correlation, and whether the
matched window lies inside the allowed time range from link.txt.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MATCH_SIZE = (210, 420)


def load_meta(seq_dir):
    meta = json.loads((seq_dir / "meta.json").read_text())
    x0, y0, x1, y1 = meta["crop_in_source"]
    return meta, (x0, y0, x1, y1)


def to_match_gray(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.resize(gray, MATCH_SIZE, interpolation=cv2.INTER_AREA)


def normalize(gray):
    flat = gray - gray.mean()
    norm = np.linalg.norm(flat)
    return flat / norm if norm > 0 else flat


def load_seq_frames(seq_dir):
    frames = []
    for path in sorted(seq_dir.glob("frame_*.png")):
        bgr = cv2.imread(str(path))
        frames.append((path.name, normalize(to_match_gray(bgr))))
    return frames


def load_video_frames(video_path, crop):
    x0, y0, x1, y1 = crop
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(normalize(to_match_gray(bgr[y0:y1, x0:x1])))
    cap.release()
    return fps, frames


def match_sequence(seq_frames, video_frames, allowed, fps):
    stack = np.stack([f.ravel() for f in video_frames])
    first = max(0, int(allowed[0] * fps))
    end = min(len(video_frames), int(np.ceil(allowed[1] * fps)) + 1)
    results = []
    for name, ref in seq_frames:
        scores = stack @ ref.ravel()
        best = first + int(np.argmax(scores[first:end]))
        global_best = int(np.argmax(scores))
        results.append(
            {
                "frame": name,
                "video_frame": best,
                "corr": float(scores[best]),
                "global_best_frame": global_best,
                "global_best_corr": float(scores[global_best]),
            }
        )
    return results


def summarize(results, fps, allowed, expected_indices):
    times = [r["video_frame"] / fps for r in results]
    corrs = [r["corr"] for r in results]
    in_allowed = [allowed[0] <= t <= allowed[1] for t in times]
    outside_better = sum(
        1
        for r in results
        if r["global_best_frame"] != r["video_frame"]
        and r["global_best_corr"] > r["corr"] + 0.02
    )
    steps = np.diff([r["video_frame"] for r in results])
    expected_hits = sum(
        1
        for r, exp in zip(results, expected_indices or [])
        if abs(r["video_frame"] - exp) <= 1
    )
    return {
        "corr_min": min(corrs),
        "corr_median": float(np.median(corrs)),
        "matched_time_range_s": [min(times), max(times)],
        "frames_in_allowed_range": sum(in_allowed),
        "frames_total": len(results),
        "monotonic": bool(np.all(steps >= 0)),
        "median_step_frames": float(np.median(steps)) if len(steps) else 0.0,
        "matches_meta_indices": expected_hits,
        "clearly_better_outside_allowed": outside_better,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--seq", type=Path, default=Path("assets/textures/flames/pillar_ref_seq")
    )
    parser.add_argument("--allowed", default="0,2")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    allowed = tuple(float(v) for v in args.allowed.split(","))
    meta, crop = load_meta(args.seq)
    seq_frames = load_seq_frames(args.seq)
    fps, video_frames = load_video_frames(args.video, crop)
    print(f"video: {len(video_frames)} frames @ {fps:.2f} fps, crop {crop}")
    print(f"sequence: {len(seq_frames)} frames from {args.seq}")

    results = match_sequence(seq_frames, video_frames, allowed, fps)
    summary = summarize(results, fps, allowed, meta.get("source_frame_indices"))

    for r in results:
        t = r["video_frame"] / fps
        inside = "in" if allowed[0] <= t <= allowed[1] else "OUT"
        print(
            f"{r['frame']}: video frame {r['video_frame']:4d} ({t:6.2f} s, {inside})"
            f"  corr {r['corr']:.3f}"
        )

    print()
    print(f"corr min/median: {summary['corr_min']:.3f} / {summary['corr_median']:.3f}")
    print(
        "matched time range: "
        f"{summary['matched_time_range_s'][0]:.2f}-{summary['matched_time_range_s'][1]:.2f} s"
        f" (allowed {allowed[0]:.0f}-{allowed[1]:.0f} s)"
    )
    print(
        f"frames inside allowed range: {summary['frames_in_allowed_range']}"
        f"/{summary['frames_total']}"
    )
    print(
        f"alignment monotonic: {summary['monotonic']},"
        f" median step {summary['median_step_frames']:.1f} video frames,"
        f" meta index hits {summary['matches_meta_indices']}/{summary['frames_total']}"
    )
    print(
        "frames matching clearly better outside the allowed range: "
        f"{summary['clearly_better_outside_allowed']}"
        " (the video replays the opening clip during the commentary)"
    )

    if args.json:
        args.json.write_text(json.dumps({"results": results, "summary": summary}, indent=1))
        print(f"json written to {args.json}")


if __name__ == "__main__":
    main()
