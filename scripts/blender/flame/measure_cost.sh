#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="$REPO_ROOT/log/blender_flame_probe/flame_cost.json"
PEAK_TFLOPS=""
GPU="nvidia"
BACKEND="vulkan"
AMD_PCI_ID="${THYLLORE_AMD_PCI_ID:-1002:744c}"
EXTRA_ARGS=()

usage() {
    cat <<USAGE
Usage: $0 --peak-tflops N [--gpu nvidia|amd] [--out PATH] [-- measure_cost.py args]

Measures the flame resolve pass GPU cost in headless Blender (docker, Vulkan,
xvfb) and writes JSON + a markdown table for the distribution spec.

  --backend vulkan|opengl
                    Blender GPU backend (default vulkan)
  --gpu nvidia|amd  nvidia (default) uses the NVIDIA container runtime; amd uses
                    Mesa radv via /dev/dri and picks the device THYLLORE_AMD_PCI_ID
                    (default $AMD_PCI_ID = RX 7900 XTX) through MESA_VK_DEVICE_SELECT

  --peak-tflops N   FP32 TFLOPS of this machine's GPU (cores x 2 x boost clock).
                    Required: converts ms into "TFLOPS needed at target fps".
  --out PATH        JSON output (default: $OUT). The .md table sits next to it.
  --                Remaining args go to measure_cost.py
                    (--preset, --resolutions, --frames, --target-fps, --camera-distance)
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --peak-tflops) PEAK_TFLOPS="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ -z "$PEAK_TFLOPS" ]]; then
    echo "--peak-tflops is required" >&2
    usage >&2
    exit 2
fi

cd "$REPO_ROOT"
if ! docker image inspect thyllore-blender-xvfb:local >/dev/null 2>&1; then
    docker build -f blender/docker/Dockerfile.xvfb -t thyllore-blender-xvfb:local blender/docker
fi
case "$GPU" in
    nvidia)
        IMAGE="thyllore-blender-xvfb:local"
        DOCKER_GPU=(--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all)
        ;;
    amd)
        IMAGE="thyllore-blender-xvfb-amd:local"
        if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
            docker build -f blender/docker/Dockerfile.xvfb-amd -t "$IMAGE" blender/docker
        fi
        DOCKER_GPU=(--device /dev/dri --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" -e "MESA_VK_DEVICE_SELECT=$AMD_PCI_ID!")
        ;;
    *) echo "invalid --gpu: $GPU" >&2; exit 2 ;;
esac
python3 scripts/blender/flame/export_glsl.py --repo-root "$REPO_ROOT" --out "$REPO_ROOT/blender_addon/effects/flame/shaders" >/dev/null
if ! ls blender_addon/effects/flame/wheels/thyllore_effect_core-*.whl >/dev/null 2>&1; then
    bash scripts/collect_wheels.sh --crate thyllore-effect-core --wheels-dir blender_addon/effects/flame/wheels
fi

mkdir -p "$(dirname "$OUT")"
LOG="${OUT%.json}.log"
docker run --rm "${DOCKER_GPU[@]}" -v "$REPO_ROOT:$REPO_ROOT" -w "$REPO_ROOT" "$IMAGE" \
    sh -c "xvfb-run -a -s '-screen 0 1280x720x24' blender --gpu-backend $BACKEND -noaudio --python-exit-code 1 \
        --python '$REPO_ROOT/scripts/blender/flame/measure_cost.py' -- --out '$OUT' --peak-tflops '$PEAK_TFLOPS' ${EXTRA_ARGS[*]:-} > '$LOG' 2>&1" || true
grep -E "^COST|Traceback|Error" "$LOG" || true
grep -q "^COST_DONE" "$LOG"

python3 - "$OUT" <<'PY'
import json, sys
path = sys.argv[1]
r = json.load(open(path))
lines = [f"# Flame rendering cost ({r['preset']}, {r['gpu']}, {r['backend']}, Blender {r['blender']})", "",
         f"peak FP32 {r['peak_tflops']} TFLOPS, {r['frames']} frames/sample, target {r['target_fps']:g} fps", "",
         "| Resolution | Scissor px | GPU ms/frame | ms/Mpx (scissor) | ms/Mpx (frame) | TFLOPS needed @ target | Minimum known GPU |",
         "|---|---|---|---|---|---|---|"]
for e in r["results"]:
    lines.append(f"| {e['resolution']} | {e['scissor_pixels']} | {e['gpu_ms_per_frame']:.2f} | {e['ms_per_mpixel_scissor'] or 0:.2f} | {e['ms_per_mpixel_frame']:.2f} | {e['required_tflops_at_target_fps']:.1f} | {e['minimum_known_gpu']} |")
md = path[:-5] + ".md"
open(md, "w").write("\n".join(lines) + "\n")
print(f"[measure_cost] wrote {path} and {md}")
PY
