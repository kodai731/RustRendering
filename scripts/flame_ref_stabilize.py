"""Remove the camera motion from the pillar_ref_seq source clip.

usage:
  python scripts/flame_ref_stabilize.py [--video /tmp/thyllore_flame_ref/pillar_src.mp4]
      [--out assets/textures/flames/pillar_ref_seq_stable] [--start 0] [--seconds 2]

Tracks ground features at the flame's depth (the band around the contact
line, flame column excluded; the camera dollies forward so the near foreground
scales faster than the flame plane and must not be used) between consecutive
frames, fits a similarity transform per step, accumulates it, smooths the
camera path with a low-order polynomial and warps every frame into the first
frame's coordinates. The residual drift of the flame foot (trunk centre above
the contact line, contact row) is then removed with a smoothed translation so
the flame always stands where it stands in frame 0. The caption overlay is
inpainted away on the caption frames. The output keeps the
frame-0 crop window of flame_ref_extract.py (uncovered areas are black),
is background-attenuated the same way and written as frame_NN.png plus
meta.json. Re-running overwrites the same files.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

import flame_ref_extract as extract

GROUND_BAND = (1050, 1400)
FLAME_COLUMN = (330, 780)
HIGH_PASS_SIGMA = 6.0
MAX_FEATURES = 800
FEATURE_QUALITY = 0.003
FEATURE_MIN_DISTANCE = 8
LK_WINDOW = (41, 41)
LK_LEVELS = 4
RANSAC_THRESHOLD = 1.5
CAMERA_PATH_POLY_DEGREE = 3
FOOT_SEARCH_ROWS = (1250, 1600)
TRUNK_ROWS = (1150, 1300)
FOOT_LUM = 150
FOOT_POLY_DEGREE = 2
CAPTION_ROWS = (800, 1200)
CAPTION_FILL_MAX_SATURATION = 70
CAPTION_FILL_MIN_VALUE = 150
CAPTION_YELLOW_HUE = (22, 32)
CAPTION_YELLOW_MIN_SATURATION = 190
CAPTION_YELLOW_MIN_VALUE = 190
CAPTION_OUTLINE_MAX_LUM = 50
CAPTION_MIN_GLYPH_AREA = 40
CAPTION_OUTLINE_REACH = 25
CAPTION_INPAINT_RADIUS = 15
CAPTION_MASK_MARGIN = 13


def read_frames(video_path, start_seconds, seconds):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = int(round(start_seconds * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    for _ in range(int(round(seconds * fps))):
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    return fps, first, frames


def high_pass_gray(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), HIGH_PASS_SIGMA)
    return np.clip(detail * 4.0 + 128.0, 0, 255).astype(np.uint8)


def ground_mask(shape):
    mask = np.zeros(shape[:2], np.uint8)
    mask[GROUND_BAND[0]:GROUND_BAND[1], :] = 255
    mask[:, FLAME_COLUMN[0]:FLAME_COLUMN[1]] = 0
    return mask


def step_transform(prev_gray, next_gray, mask):
    points = cv2.goodFeaturesToTrack(prev_gray, MAX_FEATURES, FEATURE_QUALITY, FEATURE_MIN_DISTANCE, mask=mask)
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, next_gray, points, None, winSize=LK_WINDOW, maxLevel=LK_LEVELS)
    tracked = status.ravel() == 1
    matrix, inliers = cv2.estimateAffinePartial2D(
        points[tracked], moved[tracked], method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD)
    if matrix is None:
        raise SystemExit("similarity fit failed")
    return np.vstack([matrix, [0.0, 0.0, 1.0]]), int(inliers.sum())


def cumulative_transforms(frames):
    mask = ground_mask(frames[0].shape)
    grays = [high_pass_gray(f) for f in frames]
    transforms = [np.eye(3)]
    inlier_counts = []
    for prev_gray, next_gray in zip(grays, grays[1:]):
        step, inliers = step_transform(prev_gray, next_gray, mask)
        transforms.append(step @ transforms[-1])
        inlier_counts.append(inliers)
    return smooth_camera_path(transforms), inlier_counts


def smooth_camera_path(transforms):
    frame_index = np.arange(len(transforms))
    log_scale = np.log([np.hypot(t[0, 0], t[0, 1]) for t in transforms])
    angle = np.array([np.arctan2(t[0, 1], t[0, 0]) for t in transforms])
    shift_x = np.array([t[0, 2] for t in transforms])
    shift_y = np.array([t[1, 2] for t in transforms])

    def fit(values):
        return np.polyval(np.polyfit(frame_index, values, CAMERA_PATH_POLY_DEGREE), frame_index)

    smoothed = []
    for s, a, tx, ty in zip(np.exp(fit(log_scale)), fit(angle), fit(shift_x), fit(shift_y)):
        c, sn = np.cos(a), np.sin(a)
        smoothed.append(np.array([[s * c, -s * sn, tx], [s * sn, s * c, ty], [0.0, 0.0, 1.0]]))
    return smoothed


def warp_to_first(frame, transform):
    height, width = frame.shape[:2]
    inverse = np.linalg.inv(transform)[:2]
    return cv2.warpAffine(frame, inverse, (width, height), flags=cv2.INTER_LINEAR)


def flame_foot(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    trunk = gray[TRUNK_ROWS[0]:TRUNK_ROWS[1]]
    weights = np.where(trunk > FOOT_LUM, trunk, 0.0)
    columns = np.arange(trunk.shape[1])
    centre_x = float((weights * columns).sum() / max(weights.sum(), 1.0))

    foot = gray[FOOT_SEARCH_ROWS[0]:FOOT_SEARCH_ROWS[1]]
    widths = (foot > FOOT_LUM).sum(axis=1)
    widest = int(widths.argmax())
    contact = widest + int(np.where(widths[widest:] > 0.5 * widths[widest])[0].max())
    return centre_x, float(FOOT_SEARCH_ROWS[0] + contact)


def caption_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    fill = (saturation < CAPTION_FILL_MAX_SATURATION) & (value > CAPTION_FILL_MIN_VALUE)
    yellow = ((hue >= CAPTION_YELLOW_HUE[0]) & (hue <= CAPTION_YELLOW_HUE[1])
              & (saturation > CAPTION_YELLOW_MIN_SATURATION) & (value > CAPTION_YELLOW_MIN_VALUE))
    glyphs = np.zeros(lum.shape, np.uint8)
    glyphs[CAPTION_ROWS[0]:CAPTION_ROWS[1]] = (fill | yellow)[CAPTION_ROWS[0]:CAPTION_ROWS[1]]
    glyphs = cv2.morphologyEx(glyphs, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    labels, count = ndimage.label(glyphs)
    areas = ndimage.sum(glyphs, labels, range(1, count + 1))
    glyphs = np.isin(labels, [i + 1 for i, area in enumerate(areas) if area > CAPTION_MIN_GLYPH_AREA])

    reach = np.ones((CAPTION_OUTLINE_REACH, CAPTION_OUTLINE_REACH), np.uint8)
    near_glyph = cv2.dilate(glyphs.astype(np.uint8), reach).astype(bool)
    outline = (lum < CAPTION_OUTLINE_MAX_LUM) & near_glyph
    mask = (glyphs | outline).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    return cv2.dilate(mask, np.ones((CAPTION_MASK_MARGIN, CAPTION_MASK_MARGIN), np.uint8))


def remove_caption(bgr):
    return cv2.inpaint(bgr, caption_mask(bgr), CAPTION_INPAINT_RADIUS, cv2.INPAINT_NS)


def foot_anchored_transforms(frames, transforms):
    feet = np.array([flame_foot(warp_to_first(f, t)) for f, t in zip(frames, transforms)])
    frame_index = np.arange(len(frames))
    smooth_x = np.polyval(np.polyfit(frame_index, feet[:, 0], FOOT_POLY_DEGREE), frame_index)
    smooth_y = np.polyval(np.polyfit(frame_index, feet[:, 1], FOOT_POLY_DEGREE), frame_index)

    anchored = []
    for t, x, y in zip(transforms, smooth_x, smooth_y):
        shift = np.array([[1.0, 0.0, x - smooth_x[0]], [0.0, 1.0, y - smooth_y[0]], [0.0, 0.0, 1.0]])
        anchored.append(shift @ t)
    return anchored, feet


def write_sequence(frames, transforms, out_dir, fps, first, video_path, inlier_counts, feet):
    x0, y0, x1, y1 = extract.CROP_IN_SOURCE
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.png"):
        old.unlink()
    for n, (frame, transform) in enumerate(zip(frames, transforms)):
        stable = warp_to_first(frame, transform)[y0:y1, x0:x1]
        if n in extract.CAPTION_FRAMES:
            stable = remove_caption(stable)
        cv2.imwrite(str(out_dir / f"frame_{n:02d}.png"), extract.attenuate_background(stable))

    meta = {
        "source": "youtube shorts -LzPOERYBBA (usable range 0:00-0:02 per link.txt), camera motion removed",
        "source_video": video_path.name,
        "source_resolution": [frames[0].shape[1], frames[0].shape[0]],
        "frames": len(frames),
        "fps": fps,
        "source_frame_indices": [first + n for n in range(len(frames))],
        "caption_frames": [],
        "caption_removed_frames": [i for i in extract.CAPTION_FRAMES if i < len(frames)],
        "crop_in_source": [x0, y0, x1, y1],
        "stabilization": "LK tracking of ground features at the flame depth + RANSAC similarity per step, "
                         f"accumulated to frame 0 coordinates, camera path smoothed (poly degree {CAMERA_PATH_POLY_DEGREE}), "
                         f"then the flame foot pinned to its frame-0 position (poly degree {FOOT_POLY_DEGREE}); uncovered area is black; "
                         "caption overlay inpainted on caption frames",
        "transform_to_frame0": [t[:2].round(4).tolist() for t in transforms],
        "flame_foot_before_pinning": feet.round(1).tolist(),
        "tracking_inliers": inlier_counts,
        "processing": "luminance soft-mask background attenuation "
                      f"(floor {extract.MASK_FLOOR}, smoothstep lum {extract.MASK_LUM_LOW:.0f}-{extract.MASK_LUM_HIGH:.0f}, "
                      f"gaussian blur sigma {extract.MASK_BLUR_SIGMA:.0f}px)",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("/tmp/thyllore_flame_ref/pillar_src.mp4"))
    parser.add_argument("--out", type=Path, default=Path("assets/textures/flames/pillar_ref_seq_stable"))
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    if not args.video.exists():
        extract.download(extract.read_link(Path("assets/textures/flames/pillar_ref_seq")), args.video)
    fps, first, frames = read_frames(args.video, args.start, args.seconds)
    camera, inlier_counts = cumulative_transforms(frames)
    transforms, feet = foot_anchored_transforms(frames, camera)
    write_sequence(frames, transforms, args.out, fps, first, args.video, inlier_counts, feet)

    last = camera[-1]
    drift = feet[-1] - feet[0]
    print(f"wrote {len(frames)} frames to {args.out}, camera zoom {np.hypot(last[0, 0], last[0, 1]):.3f}x, "
          f"min inliers {min(inlier_counts)}, foot drift removed ({drift[0]:+.0f}, {drift[1]:+.0f}) px")


if __name__ == "__main__":
    main()
