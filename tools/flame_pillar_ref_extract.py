"""Extract the pillar_ref_seq reference frames from the 1080x1920 source video.

Usage:
  python tools/flame_pillar_ref_extract.py --video <pillar_src_1080.mp4> --out assets/textures/flames/pillar_ref_seq
    [--match-dir assets/textures/flames/pillar_ref_seq/low_quality] [--scale 3] [--stride 3]

`--stride 1` writes every source frame of the same span (30 fps, for the sequence gates xiii-xvi of
scripts/flame_ref_match.py); the default stride 3 reproduces the legacy 10 fps frames.

The 40 output frames reproduce the legacy 280x560 sequence (now in low_quality/) at
`scale` times the resolution: the same crop of the source, the same frames (the first found by
luminance correlation against the legacy frames, then every third source frame), and the same background attenuation
(floor + (1 - floor) * blur(smoothstep(lum, lo, hi))) with the blur radius scaled.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

LEGACY_SOURCE_SIZE = (360, 640)
LEGACY_CROP = (40, 0, 320, 560)
LEGACY_BLUR_SIGMA = 6.0
MASK_LUM_LOW = 70.0
MASK_LUM_HIGH = 210.0
BACKGROUND_FLOOR = 0.12
SOURCE_FRAMES_PER_LEGACY_FRAME = 3


def luminance(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def smoothstep(x, low, high):
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def attenuate_background(rgb, blur_sigma):
    mask = ndimage.gaussian_filter(smoothstep(luminance(rgb), MASK_LUM_LOW, MASK_LUM_HIGH), blur_sigma)
    gain = BACKGROUND_FLOOR + (1.0 - BACKGROUND_FLOOR) * mask
    return np.clip(rgb * gain[..., None], 0.0, 255.0)


def read_source_frames(video_path):
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(bgr[:, :, ::-1])
    return frames


def legacy_view(rgb):
    small = cv2.resize(rgb, LEGACY_SOURCE_SIZE, interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = LEGACY_CROP
    return luminance(small[y0:y1, x0:x1].astype(np.float32))


def load_legacy_frames(match_dir):
    paths = sorted(Path(match_dir).glob("frame_*.png"))
    return [luminance(np.asarray(Image.open(p).convert("RGB")).astype(np.float32)) for p in paths]


def masked_correlation(a, b, mask):
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.corrcoef(a[mask], b[mask])[0, 1]
    return -1.0 if np.isnan(score) else float(score)


def find_first_source_index(source_frames, legacy_first):
    mask = legacy_first > 90
    scores = [masked_correlation(legacy_view(f), legacy_first, mask) for f in source_frames]
    return int(np.argmax(scores))


def match_source_indices(source_frames, legacy_frames, stride):
    first = find_first_source_index(source_frames, legacy_frames[0])
    last = first + (len(legacy_frames) - 1) * SOURCE_FRAMES_PER_LEGACY_FRAME
    indices = []
    for index in range(first, last + 1, stride):
        legacy_k, remainder = divmod(index - first, SOURCE_FRAMES_PER_LEGACY_FRAME)
        score = float("nan")
        if remainder == 0:
            legacy = legacy_frames[legacy_k]
            score = masked_correlation(legacy_view(source_frames[index]), legacy, legacy > 90)
        indices.append((index, score))
    return indices


def scaled_crop(scale):
    return tuple(v * scale for v in LEGACY_CROP)


def extract_frame(rgb, scale):
    target = (LEGACY_SOURCE_SIZE[0] * scale, LEGACY_SOURCE_SIZE[1] * scale)
    resized = cv2.resize(rgb, target, interpolation=cv2.INTER_AREA) if rgb.shape[1] != target[0] else rgb
    x0, y0, x1, y1 = scaled_crop(scale)
    return attenuate_background(resized[y0:y1, x0:x1].astype(np.float32), LEGACY_BLUR_SIGMA * scale)


def write_meta(out_dir, scale, indices, video_path, fps, stride):
    x0, y0, x1, y1 = scaled_crop(scale)
    meta = {
        "source": "youtube shorts -LzPOERYBBA (design 20260811_flame_noise_motion/reference)",
        "source_video": Path(video_path).name,
        "source_resolution": [LEGACY_SOURCE_SIZE[0] * scale, LEGACY_SOURCE_SIZE[1] * scale],
        "frames": len(indices),
        "fps": fps / stride,
        "source_frame_indices": [i for i, _ in indices],
        "legacy_match_correlation_min": min(s for _, s in indices if s == s),
        "crop_in_source": [x0, y0, x1, y1],
        "processing": (
            f"luminance soft-mask background attenuation (floor {BACKGROUND_FLOOR}, "
            f"smoothstep lum {MASK_LUM_LOW:.0f}-{MASK_LUM_HIGH:.0f}, gaussian blur sigma {LEGACY_BLUR_SIGMA * scale:.0f}px)"
        ),
        "legacy": "low_quality/ holds the previous 280x560 sequence this one reproduces at higher resolution",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--match-dir", default=None, help="legacy frames to align to (default: <out>/low_quality)")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--stride", type=int, default=SOURCE_FRAMES_PER_LEGACY_FRAME)
    args = parser.parse_args()

    out_dir = Path(args.out)
    match_dir = Path(args.match_dir) if args.match_dir else out_dir / "low_quality"
    source_frames = read_source_frames(args.video)
    fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS)
    legacy_frames = load_legacy_frames(match_dir)
    indices = match_source_indices(source_frames, legacy_frames, args.stride)

    out_dir.mkdir(parents=True, exist_ok=True)
    digits = len(str(len(indices) - 1))
    for k, (index, score) in enumerate(indices):
        frame = extract_frame(source_frames[index], args.scale)
        name = f"frame_{k:0{digits}d}"
        Image.fromarray(frame.astype(np.uint8)).save(out_dir / f"{name}.png")
        print(f"{name} <- source {index} (corr {score:.3f})")
    write_meta(out_dir, args.scale, indices, args.video, fps, args.stride)


if __name__ == "__main__":
    main()
