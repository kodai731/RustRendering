#!/usr/bin/env bash
# SSoT GPU wiring for Blender.
#
# Pins every Blender GPU launch to the RX 550 (card3, PCI 0000:44:00.0, 1002:67ff)
# so the RX 7900 XTX (card2, gfx1100) keeps its full VRAM for codeagent training.
#
# This wrapper is the single source: .claude/local/paths.md BlenderPath points
# here, and the interactive shell alias / desktop entry calls the same file.
#
# NOTE: The NVIDIA Blackwell (card1) has no userspace driver installed
# (kernel module only), so it cannot host Blender yet. Once the NVIDIA
# proprietary + CUDA userspace is installed, switch the exports below to:
#   export CUDA_VISIBLE_DEVICES=0
#   export __NV_PRIME_RENDER_OFFLOAD=1
#   export __GLX_VENDOR_LIBRARY_NAME=nvidia
#   export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
set -euo pipefail

export DRI_PRIME=pci-0000_44_00_0
export MESA_VK_DEVICE_SELECT=1002:67ff

exec /opt/blender/blender "$@"
