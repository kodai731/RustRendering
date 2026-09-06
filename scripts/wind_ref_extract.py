"""Extract the wind tornado reference sequences for scripts/flame_ref_match.py style gates.

usage:
  python scripts/wind_ref_extract.py castle [--video /tmp/thyllore_wind_ref/castle_src.mp4]
      [--out assets/textures/wind/castle_ref_seq]
  python scripts/wind_ref_extract.py storm [--video /tmp/thyllore_wind_ref/storm_src.mp4]
      [--out assets/textures/wind/storm_ref_seq] [--start 15] [--end 55]

Downloads the clip named in <out>/link.txt with yt-dlp when --video is absent and writes
frame_NNN.png (dust on a black background, source colours kept, storm frames at half size) plus meta.json.

castle  MH4G Kushala Daora intro, 32.45-34.30 s, one shot. The dust haze cannot be told from the
        blown-out sky by colour, so the mask is the temporal flicker: consecutive frames are aligned
        with SIFT homographies chained around each frame, and the std of min(R,G,B) over the window,
        gated by whiteness, attenuates everything that does not swirl.
storm   MH Wilds Kushala Daora storm, 15-55 s, many shots. Ground, hunters and sky are dark or
        saturated, so the mask is the colour alone: high HSV value and low saturation. Shots are split
        at HSV histogram cuts and written as one sub-directory each.

Re-running overwrites the same files.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CASTLE = {
    "source": "youtube xFR4gWP8g98 (MH4G Kushala Daora intro, 32.45-34.30 s, one shot)",
    "format": "298",
    "start": 32.45,
    "end": 34.30,
    "crop": (320, 0, 960, 720),
    "output_scale": 1.0,
}
STORM = {
    "source": "youtube YaZpzzrYDNA (MH Wilds Kushala Daora storm, 15-55 s, split per shot)",
    "format": "136",
    "start": 15.0,
    "end": 55.0,
    "crop": (0, 0, 1280, 720),
    "output_scale": 0.5,
}

FLICKER_WINDOW_HALF = 4
FLICKER_STD_LOW, FLICKER_STD_HIGH = 2.5, 10.0
FLICKER_WHITE_FLOOR = 200.0
FLICKER_BLUR = 25
FLICKER_OPEN = 15
MIN_INLIERS = 30

VALUE_LOW, VALUE_HIGH = 110.0, 235.0
SATURATION_FULL, SATURATION_ZERO = 35.0, 95.0
STORM_BLUR = 15
STORM_OPEN = 9
CUT_THRESHOLD = 0.22
MIN_SHOT_FRAMES = 6


def read_link(out_dir):
    for line in (out_dir / "link.txt").read_text().splitlines():
        if line.startswith("http"):
            return line.strip()
    raise SystemExit("link.txt has no http line")


def download(url, video_path, format_id):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-q", "-f", format_id, "-o", str(video_path), "--force-overwrites", url],
        check=True,
    )


def read_frames(video_path, start_seconds, end_seconds):
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first = int(round(start_seconds * fps))
    last = int(round(end_seconds * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    for _ in range(first, last + 1):
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(bgr)
    capture.release()
    return fps, (width, height), first, frames


def ramp(x, low, high):
    return np.clip((x - low) / (high - low), 0.0, 1.0)


def open_mask(mask, kernel):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)


def apply_mask(bgr, mask):
    return np.clip(bgr.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def write_sequence(out_dir, frames, masks, crop, output_scale, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.png"):
        old.unlink()
    x0, y0, x1, y1 = crop
    for index, (bgr, mask) in enumerate(zip(frames, masks)):
        cropped = apply_mask(bgr, mask)[y0:y1, x0:x1]
        if output_scale != 1.0:
            cropped = cv2.resize(cropped, None, fx=output_scale, fy=output_scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"frame_{index:03d}.png"), cropped)
    meta["frames"] = len(frames)
    meta["caption_frames"] = []
    meta["crop_in_source"] = list(crop)
    meta["output_scale"] = output_scale
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def homography_between(reference_gray, moving_gray):
    sift = cv2.SIFT_create(3000)
    ref_kp, ref_desc = sift.detectAndCompute(reference_gray, None)
    mov_kp, mov_desc = sift.detectAndCompute(moving_gray, None)
    if ref_desc is None or mov_desc is None or len(ref_kp) < 12 or len(mov_kp) < 12:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    good = [m for m, n in matcher.knnMatch(mov_desc, ref_desc, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None
    src = np.float32([mov_kp[m.queryIdx].pt for m in good])
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good])
    homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if homography is None or int(inliers.sum()) < MIN_INLIERS:
        return None
    return homography


def consecutive_homographies(grays):
    """step[i] maps frame i+1 into frame i coordinates; None where the alignment failed."""
    return [homography_between(grays[i], grays[i + 1]) for i in range(len(grays) - 1)]


def chain_to(steps, center, other):
    """Homography mapping frame `other` into frame `center` coordinates by composing the steps."""
    total = np.eye(3)
    if other > center:
        for i in range(center, other):
            if steps[i] is None:
                return None
            total = total @ steps[i]
    else:
        for i in range(other, center):
            if steps[i] is None:
                return None
            total = total @ np.linalg.inv(steps[i])
    return total


def flicker_masks(frames):
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    neutral = [f.min(axis=2).astype(np.float32) for f in frames]
    height, width = grays[0].shape
    steps = consecutive_homographies(grays)
    print(f"aligned {sum(s is not None for s in steps)}/{len(steps)} consecutive pairs")

    masks = []
    for center in range(len(frames)):
        stack = [neutral[center]]
        for other in range(max(0, center - FLICKER_WINDOW_HALF), min(len(frames), center + FLICKER_WINDOW_HALF + 1)):
            if other == center:
                continue
            homography = chain_to(steps, center, other)
            if homography is None:
                continue
            stack.append(cv2.warpPerspective(neutral[other], homography, (width, height), borderMode=cv2.BORDER_REPLICATE))
        temporal_std = cv2.GaussianBlur(np.std(np.stack(stack), axis=0), (FLICKER_BLUR, FLICKER_BLUR), 0)
        flicker = ramp(temporal_std, FLICKER_STD_LOW, FLICKER_STD_HIGH)
        white = ramp(cv2.GaussianBlur(neutral[center], (7, 7), 0), FLICKER_WHITE_FLOOR, 255.0)
        masks.append(open_mask(flicker * white, FLICKER_OPEN))
    return masks


def dust_colour_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = cv2.GaussianBlur(hsv[..., 1], (STORM_BLUR, STORM_BLUR), 0)
    value = cv2.GaussianBlur(hsv[..., 2], (7, 7), 0)
    mask = ramp(value, VALUE_LOW, VALUE_HIGH) * (1.0 - ramp(saturation, SATURATION_FULL, SATURATION_ZERO))
    return open_mask(mask, STORM_OPEN)


def histogram(bgr):
    small = cv2.resize(bgr, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, None).flatten()


def split_shots(frames):
    """Index ranges [first, end) between HSV histogram cuts."""
    histograms = [histogram(f) for f in frames]
    boundaries = [0]
    for i in range(1, len(frames)):
        a, b = histograms[i - 1], histograms[i]
        distance = 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        if distance > CUT_THRESHOLD:
            boundaries.append(i)
    boundaries.append(len(frames))
    return [(a, b) for a, b in zip(boundaries[:-1], boundaries[1:]) if b - a >= MIN_SHOT_FRAMES]


def base_meta(config, video_path, resolution, fps, first_index, count):
    return {
        "source": config["source"],
        "source_video": video_path.name,
        "source_resolution": list(resolution),
        "fps": fps,
        "source_frame_indices": list(range(first_index, first_index + count)),
    }


def extract_castle(video_path, out_dir):
    fps, resolution, first, frames = read_frames(video_path, CASTLE["start"], CASTLE["end"])
    masks = flicker_masks(frames)
    meta = base_meta(CASTLE, video_path, resolution, fps, first, len(frames))
    meta["processing"] = (
        f"temporal flicker mask: std of min(R,G,B) over +-{FLICKER_WINDOW_HALF} SIFT-aligned frames "
        f"ramp {FLICKER_STD_LOW}-{FLICKER_STD_HIGH}, times whiteness ramp {FLICKER_WHITE_FLOOR:.0f}-255, "
        f"gaussian {FLICKER_BLUR}px, opening {FLICKER_OPEN}px, background floor 0"
    )
    write_sequence(out_dir, frames, masks, CASTLE["crop"], CASTLE["output_scale"], meta)
    print(f"wrote {len(frames)} frames at {fps:.3f} fps to {out_dir}")


def extract_storm(video_path, out_dir, start_seconds, end_seconds):
    fps, resolution, first, frames = read_frames(video_path, start_seconds, end_seconds)
    shots = split_shots(frames)
    for old in out_dir.glob("shot_*"):
        for file in old.iterdir():
            file.unlink()
        old.rmdir()

    index_lines = []
    for shot_number, (a, b) in enumerate(shots):
        shot_start = (first + a) / fps
        shot_end = (first + b) / fps
        shot_dir = out_dir / f"shot_{shot_number:02d}_{shot_start:05.2f}s"
        masks = [dust_colour_mask(f) for f in frames[a:b]]
        meta = base_meta(STORM, video_path, resolution, fps, first + a, b - a)
        meta["shot_seconds"] = [shot_start, shot_end]
        meta["processing"] = (
            f"colour mask: HSV value ramp {VALUE_LOW:.0f}-{VALUE_HIGH:.0f} times saturation ramp "
            f"{SATURATION_FULL:.0f}->{SATURATION_ZERO:.0f} (1->0), gaussian {STORM_BLUR}px, opening {STORM_OPEN}px, "
            f"background floor 0"
        )
        write_sequence(shot_dir, frames[a:b], masks, STORM["crop"], STORM["output_scale"], meta)
        index_lines.append(f"{shot_dir.name} {shot_start:.2f}-{shot_end:.2f}s {b - a} frames")
        print(index_lines[-1])
    (out_dir / "shots.txt").write_text("\n".join(index_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", choices=["castle", "storm"])
    parser.add_argument("--video", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--start", type=float, default=STORM["start"])
    parser.add_argument("--end", type=float, default=STORM["end"])
    args = parser.parse_args()

    config = CASTLE if args.clip == "castle" else STORM
    video_path = args.video or Path(f"/tmp/thyllore_wind_ref/{args.clip}_src.mp4")
    out_dir = args.out or Path(f"assets/textures/wind/{args.clip}_ref_seq")
    if not video_path.exists():
        download(read_link(out_dir), video_path, config["format"])

    if args.clip == "castle":
        extract_castle(video_path, out_dir)
    else:
        extract_storm(video_path, out_dir, args.start, args.end)


if __name__ == "__main__":
    main()
