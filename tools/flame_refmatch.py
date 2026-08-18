import sys
import json
import subprocess
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flame_wallness import detect_viewport_from_background, dood_wrap, engine_path, engine_env

REFERENCE_CONFIGS = {
    "pillar": {
        "reference_dir": "assets/textures/flames/pillar_ref_seq",
        "camera": "30,-8,14,0,3,0",
        "preset": "ring",
        "framing": ["height=8.0", "radius=0.5"],
        "look": [
            "noise_amplitude=3", "noise_contrast=4", "swirl_gain=1.5",
            "spread_gain=3", "support_margin=2.0", "swirl_speed=1.0", "noise_frequency=3",
            "meander_amp=0.8"
        ]
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Flame reference matching tool.")
    parser.add_argument("--out-dir", type=str, default="target/tmp_screens/refmatch",
                        help="Output directory")
    parser.add_argument("--skip-capture", action="store_true",
                        help="Skip capturing sequences and use existing ones")
    parser.add_argument("--dood", action="store_true",
                        help="Use DOOD for capture commands")
    return parser.parse_args()


def capture_background(out_dir: Path, camera: str, preset: str, out_path: Path, dood: bool = False) -> None:
    """Capture a single background frame: same preset/scene with the flame extinguished (intensity=0)."""
    cmd = [
        str(engine_path()),
        "--batch-screenshot", str(out_path),
        "--batch-camera", camera,
        "--batch-flame-preset", preset,
        "--batch-flame-mode", "analytic",
        "--batch-frames", "60",
        "--batch-flame-set", "intensity=0",
    ]
    if dood:
        cmd = dood_wrap(cmd)
    print(f"[refmatch] capture background: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=engine_env())
    last_line = result.stdout.strip().split("\n")[-1]
    report = json.loads(last_line)
    if not report.get("ok"):
        raise RuntimeError(report.get("error", "background capture failed"))


def capture_sequence(out_dir: Path, arm_name: str, camera: str, preset: str,
                     framing: list, look: list, dood: bool = False) -> None:
    """Capture a 40-frame sequence for an arm using --batch-screenshot-sequence."""
    seq_out = str(out_dir / arm_name)
    cmd = [
        str(engine_path()),
        "--batch-screenshot-sequence", f"{seq_out},40,6",
        "--batch-camera", camera,
        "--batch-flame-preset", preset,
        "--batch-flame-mode", "analytic",
        "--batch-frames", "60",
    ]
    for f in framing:
        cmd.extend(["--batch-flame-set", f])
    for l in look:
        cmd.extend(["--batch-flame-set", l])
    if dood:
        cmd = dood_wrap(cmd)
    print(f"[refmatch] capture sequence {arm_name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=engine_env())
    last_line = result.stdout.strip().split("\n")[-1]
    report = json.loads(last_line)
    if not report.get("ok"):
        raise RuntimeError(report.get("error", f"sequence capture failed for {arm_name}"))


def capture_all(out_dir: Path, config: dict, dood: bool = False) -> None:
    """Capture background + current/legacy sequences."""
    out_dir.mkdir(parents=True, exist_ok=True)
    camera = config["camera"]
    preset = config["preset"]
    framing = config["framing"]
    look_current = list(config["look"])

    # Build legacy look: replace support_margin and meander_amp
    look_legacy = []
    for item in look_current:
        if item.startswith("support_margin="):
            look_legacy.append("support_margin=1.0")
        elif item.startswith("meander_amp="):
            look_legacy.append("meander_amp=0.0")
        else:
            look_legacy.append(item)

    capture_background(out_dir, camera, preset, out_dir / "background.png", dood=dood)
    capture_sequence(out_dir, "current", camera, preset, framing, look_current, dood=dood)
    capture_sequence(out_dir, "legacy", camera, preset, framing, look_legacy, dood=dood)


def preprocess_sequences(out_dir: Path) -> None:
    """Crop frames by detected viewport and subtract background (clamp 0)."""
    from PIL import Image
    bg_path = out_dir / "background.png"
    if not bg_path.exists():
        raise FileNotFoundError("background.png not found — run capture first")

    x0, y0, x1, y1 = detect_viewport_from_background(bg_path)
    print(f"[refmatch] detected viewport: x0={x0}, y0={y0}, x1={x1}, y1={y1}")

    # Load background image once for subtraction
    bg_img = Image.open(bg_path).convert("RGB")
    bg_crop = bg_img.crop((x0, y0, x1, y1))
    bg_arr = list(bg_crop.getdata())

    for arm_name in ("current", "legacy"):
        src_dir = out_dir / arm_name
        prep_dir = out_dir / f"{arm_name}_prep"
        prep_dir.mkdir(parents=True, exist_ok=True)

        # Copy meta.json if it exists
        meta_src = src_dir / "meta.json"
        if meta_src.exists():
            shutil.copy2(meta_src, prep_dir / "meta.json")
        else:
            # Generate meta.json with fps=10
            with open(prep_dir / "meta.json", "w") as f:
                json.dump({"fps": 10}, f)

        # Process each frame
        frames = sorted(src_dir.glob("frame_*.png"))
        for frame_path in frames:
            img = Image.open(frame_path).convert("RGB")
            crop = img.crop((x0, y0, x1, y1))
            data = list(crop.getdata())

            # Subtract background brightness per pixel (clamp 0)
            new_data = []
            for i, (c_r, c_g, c_b) in enumerate(data):
                b_r, b_g, b_b = bg_arr[i]
                # Subtract brightness: compute each channel difference, clamp to 0
                nr = max(0, c_r - b_r)
                ng = max(0, c_g - b_g)
                nb = max(0, c_b - b_b)
                new_data.append((nr, ng, nb))

            crop.putdata(new_data)
            out_name = frame_path.name
            crop.save(prep_dir / out_name)

        print(f"[refmatch] preprocessed {len(frames)} frames for {arm_name}")


def run_analysis(out_dir: Path, reference_dir: str) -> dict:
    """Run engine --batch-sequence-analyze on reference (full/first-half/second-half) and arms."""
    ref_path = Path(reference_dir)
    analyses = [
        ("ref_full", str(ref_path)),
        ("ref_first", f"{str(ref_path)},0,19"),
        ("ref_second", f"{str(ref_path)},20,39"),
        ("current", str(out_dir / "current_prep")),
        ("legacy", str(out_dir / "legacy_prep")),
    ]

    dump_file = out_dir / "analysis_dump.json"
    cmd = [str(engine_path())]
    for label, path in analyses:
        cmd.extend(["--batch-sequence-analyze", path])
    cmd.extend(["--batch-sequence-dump", str(dump_file)])

    print(f"[refmatch] analysis: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=engine_env())
    if result.returncode != 0 or not dump_file.exists():
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise RuntimeError(f"analysis failed (exit {result.returncode}): {detail}")

    with open(dump_file) as f:
        results = json.load(f)

    # Map labels to descriptors by zipping ordered analysis list with results["sequences"]
    sequences = {}
    for (label, _), seq in zip(analyses, results["sequences"]):
        sequences[label] = seq["descriptors"]
    return {"sequences": [{"dir": label, "descriptors": desc} for label, desc in sequences.items()]}

def _r0_px(descriptors: dict) -> float:
    """Pillar radius in pixels, from the extractor's normalization metadata."""
    return float(descriptors["meta"]["r0_px"])


def _zeta_max(descriptors: dict) -> float:
    """Observable normalized axial extent, from the extractor's normalization metadata."""
    return float(descriptors["meta"]["zeta_max"])


def _distance_f1(a: dict, b: dict) -> float:
    """F1 distance: RMS difference of 48-point width arrays."""
    wa = a["f1_width"]
    wb = b["f1_width"]
    n = min(len(wa), len(wb))
    if n == 0:
        return 0.0
    sq_sum = sum((wa[i] - wb[i]) ** 2 for i in range(n))
    return (sq_sum / n) ** 0.5


def _distance_f2(a: dict, b: dict) -> float:
    """F2 distance: |Δf2_rough| + |Δf2_low_band_ratio|."""
    return abs(a["f2_rough"] - b["f2_rough"]) + abs(a["f2_low_band_ratio"] - b["f2_low_band_ratio"])


def _distance_f3(a: dict, b: dict) -> float:
    """F3 distance: RMS difference of 48-point flicker arrays."""
    fa = a["f3_flicker"]
    fb = b["f3_flicker"]
    n = min(len(fa), len(fb))
    if n == 0:
        return 0.0
    sq_sum = sum((fa[i] - fb[i]) ** 2 for i in range(n))
    return (sq_sum / n) ** 0.5


def _distance_f4(a: dict, b: dict) -> float:
    """F4 distance: |Δf4_meander_rms|."""
    return abs(a["f4_meander_rms"] - b["f4_meander_rms"])


def _distance_f7(a: dict, b: dict) -> float:
    """F7 distance: |Δf7_base_disconnected_ratio|."""
    return abs(a["f7_base_disconnected_ratio"] - b["f7_base_disconnected_ratio"])


DISTANCE_FUNCS = {
    "F1": _distance_f1,
    "F2": _distance_f2,
    "F3": _distance_f3,
    "F4": _distance_f4,
    "F7": _distance_f7,
}

GATED_FAMILIES = ["F1", "F2", "F3", "F4", "F7"]


def compute_distances_and_output(results: dict) -> dict:
    """Compute family distances, scores, and match verdict from analysis results."""
    sequences = {s["dir"]: s["descriptors"] for s in results["sequences"]}

    ref_full = sequences["ref_full"]
    ref_first = sequences["ref_first"]
    ref_second = sequences["ref_second"]
    current = sequences["current"]
    legacy = sequences["legacy"]

    # Reference properties
    ref_r0_px = _r0_px(ref_full)
    ref_zeta_max = _zeta_max(ref_full)
    ref_fps = ref_full["meta"]["fps"]

    # Ceilings: d_f(ref_first, ref_second) for each family
    ceilings = {}
    for fam in GATED_FAMILIES:
        c = DISTANCE_FUNCS[fam](ref_first, ref_second)
        ceilings[fam] = round(c, 6)

    def arm_metrics(arm_descriptors: dict) -> dict:
        """Compute d_hat, score, gap_closed for each family against reference."""
        families = {}
        for fam in GATED_FAMILIES:
            d = DISTANCE_FUNCS[fam](arm_descriptors, ref_full)
            c = max(ceilings[fam], 1e-6)
            d_hat = d / c
            score = 1.0 / (1.0 + max(0.0, d_hat - 1.0))
            families[fam] = {
                "d": round(d, 6),
                "d_hat": round(d_hat, 6),
                "score": round(score, 6),
            }
        return families

    current_families = arm_metrics(current)
    legacy_families = arm_metrics(legacy)

    # gap_closed: (d_hat_legacy - d_hat_current) / max(d_hat_legacy - 1, 1e-6)
    for fam in GATED_FAMILIES:
        d_hat_legacy = legacy_families[fam]["d_hat"]
        d_hat_current = current_families[fam]["d_hat"]
        gap = (d_hat_legacy - d_hat_current) / max(d_hat_legacy - 1.0, 1e-6)
        current_families[fam]["gap_closed"] = round(gap, 6)

    # Match scores: mean of gated family scores
    match_current = sum(current_families[f]["score"] for f in GATED_FAMILIES) / len(GATED_FAMILIES)
    match_legacy = sum(legacy_families[f]["score"] for f in GATED_FAMILIES) / len(GATED_FAMILIES)

    # Coverage: arm zeta_max >= reference zeta_max
    coverage_current = _zeta_max(current) >= ref_zeta_max
    coverage_legacy = _zeta_max(legacy) >= ref_zeta_max

    # Pass: coverage_ok AND all gated gap_closed >= 0.5 AND match >= 0.6 AND match >= match_legacy + 0.15
    all_gaps_ok = all(current_families[f]["gap_closed"] >= 0.5 for f in GATED_FAMILIES)
    pass_current = coverage_current and all_gaps_ok and match_current >= 0.6 and match_current >= match_legacy + 0.15

    # Build output
    output = {
        "ok": True,
        "config": "pillar",
        "reference": {
            "r0_px": round(ref_r0_px, 2),
            "zeta_max": round(ref_zeta_max, 4),
            "fps": ref_fps,
            "ceiling": ceilings,
        },
        "arms": {
            "current": {
                "coverage_ok": coverage_current,
                "families": current_families,
                "match": round(match_current, 6),
                "pass": pass_current,
            },
            "legacy": {
                "coverage_ok": coverage_legacy,
                "families": legacy_families,
                "match": round(match_legacy, 6),
                "pass": False,
            },
        },
        "pass": pass_current,
    }
    return output

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    config = REFERENCE_CONFIGS["pillar"]

    if not args.skip_capture:
        capture_all(out_dir, config, dood=args.dood)
        preprocess_sequences(out_dir)

    results = run_analysis(out_dir, config["reference_dir"])
    output = compute_distances_and_output(results)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
