"""Extract the pillar_ref_seq reference frames from the source YouTube short.

usage:
  python scripts/flame_ref_extract.py [--video /tmp/thyllore_flame_ref/pillar_src.mp4]
      [--out assets/textures/flames/pillar_ref_seq] [--start 0] [--seconds 2]

Downloads the clip named in <out>/link.txt with yt-dlp when --video is absent,
crops the flame column, attenuates the background with a luminance soft mask
and writes frame_NN.png plus meta.json. Re-running overwrites the same files.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

CROP_IN_SOURCE = (120, 0, 960, 1680)
MASK_FLOOR = 0.12
MASK_LUM_LOW = 70.0
MASK_LUM_HIGH = 210.0
MASK_BLUR_SIGMA = 18.0
CAPTION_FRAMES = list(range(12, 38)) + list(range(56, 60))


def read_link(out_dir):
    for line in (out_dir / "link.txt").read_text().splitlines():
        if line.startswith("http"):
            return line.strip()
    raise SystemExit("link.txt has no http line")


def download(url, video_path):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["yt-dlp", "-q", "-f", "bv*[ext=mp4][height<=1920]/bv*[ext=mp4]/best",
         "-o", str(video_path), "--force-overwrites", url],
        check=True,
    )


def attenuate_background(bgr):
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    t = np.clip((lum - MASK_LUM_LOW) / (MASK_LUM_HIGH - MASK_LUM_LOW), 0.0, 1.0)
    mask = t * t * (3.0 - 2.0 * t)
    mask = cv2.GaussianBlur(mask, (0, 0), MASK_BLUR_SIGMA)
    weight = MASK_FLOOR + (1.0 - MASK_FLOOR) * mask
    return np.clip(bgr.astype(np.float32) * weight[..., None], 0, 255).astype(np.uint8)


def extract(video_path, out_dir, start_seconds, seconds):
    x0, y0, x1, y1 = CROP_IN_SOURCE
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first = int(round(start_seconds * fps))
    count = int(round(seconds * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    for old in out_dir.glob("frame_*.png"):
        old.unlink()
    indices = []
    for n in range(count):
        ok, bgr = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"frame_{n:02d}.png"), attenuate_background(bgr[y0:y1, x0:x1]))
        indices.append(first + n)
    cap.release()

    meta = {
        "source": "youtube shorts -LzPOERYBBA (usable range 0:00-0:02 per link.txt)",
        "source_video": video_path.name,
        "source_resolution": [width, height],
        "frames": len(indices),
        "fps": fps,
        "source_frame_indices": indices,
        "caption_frames": [i for i in CAPTION_FRAMES if i < len(indices)],
        "crop_in_source": list(CROP_IN_SOURCE),
        "processing": "luminance soft-mask background attenuation "
                      f"(floor {MASK_FLOOR}, smoothstep lum {MASK_LUM_LOW:.0f}-{MASK_LUM_HIGH:.0f}, "
                      f"gaussian blur sigma {MASK_BLUR_SIGMA:.0f}px)",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return len(indices), fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("/tmp/thyllore_flame_ref/pillar_src.mp4"))
    parser.add_argument("--out", type=Path, default=Path("assets/textures/flames/pillar_ref_seq"))
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    if not args.video.exists():
        download(read_link(args.out), args.video)
    count, fps = extract(args.video, args.out, args.start, args.seconds)
    print(f"wrote {count} frames at {fps:.3f} fps to {args.out}")


if __name__ == "__main__":
    main()
