#!/usr/bin/env bash
# SSoT: side-by-side comparison of Blender (ground truth) vs the Thyllore engine.
#
#   left  = Blender EEVEE reference, rendered on the RX 550 (blender_offload.sh)
#   right = Thyllore Vulkan render on the NVIDIA Blackwell (CDI container)
#
# Both sides share one auto-framed camera computed from the armature bounds so
# the comparison is fair. Output is a single PNG (default log/compare_v4.png).
#
# Usage:
#   scripts/compare_blender_thyllore.sh [usd] [out_png] [resolution] [frame]
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usd="${1:-blender/20260624_ellie_animation/ellie_animation.usdc}"
out="${2:-log/compare_v4.png}"
resolution="${3:-512}"
frame="${4:-1}"
blend="${5:-${usd%.usdc}.blend}"
fov_deg=39.6

blender="${repo_root}/scripts/blender_offload.sh"
blender_gt="${repo_root}/log/compare_blender.png"
engine_png="${repo_root}/log/compare_engine.png"

mkdir -p "${repo_root}/log"

echo "[compare] 1/4 auto-framing camera from ${blend}"
cam_out="$("${blender}" --background --python "${repo_root}/scripts/_cam_params.py" \
    -- "${repo_root}/${blend}" 2>/dev/null)"
read -r _ cx cy cz <<<"$(echo "${cam_out}" | grep '^CENTER')"
read -r _ px py pz <<<"$(echo "${cam_out}" | grep '^CAMPOS')"
echo "[compare]     pos=${px},${py},${pz} target=${cx},${cy},${cz} fov=${fov_deg}"

echo "[compare] 2/4 Blender ground-truth render (RX 550) -> ${blender_gt}"
config_json="$(printf '{"camera_pos":[%s,%s,%s],"camera_target":[%s,%s,%s],"fov_deg":%s,"resolution":%s,"frame":%s}' \
    "$px" "$py" "$pz" "$cx" "$cy" "$cz" "$fov_deg" "$resolution" "$frame")"
"${blender}" --background "${repo_root}/${blend}" --python "${repo_root}/scripts/blender_render_blend.py" -- \
    "${blender_gt}" "${config_json}"

echo "[compare] 3/4 Thyllore engine render (Blackwell container) -> ${engine_png}"
"${repo_root}/scripts/render_usd_blackwell.sh" \
    --render-usd "${usd}" --out "log/compare_engine.png" --gpu \
    --camera-pos "${px},${py},${pz}" --camera-target "${cx},${cy},${cz}" \
    --fov "${fov_deg}" --resolution "${resolution}" --frame "${frame}"

echo "[compare] 4/4 composing side-by-side -> ${out}"
"${blender}" --background --python "${repo_root}/scripts/_compose_compare.py" -- \
    "${blender_gt}" "${engine_png}" "${repo_root}/${out}"

echo "[compare] done: ${out}"
