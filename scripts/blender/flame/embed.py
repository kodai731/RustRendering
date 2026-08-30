import os
from pathlib import Path

import bpy

SCREENSHOT_DIR = os.environ.get("THYLLORE_SCREENSHOT_DIR", "")
SCREENSHOT_DELAY_SECONDS = float(os.environ.get("THYLLORE_SCREENSHOT_DELAY", "3"))
QUIT_AFTER_SCREENSHOT = os.environ.get("THYLLORE_SCREENSHOT_QUIT", "") == "1"


def add_campfire():
    existing = [o.name for o in bpy.context.scene.objects if o.thyllore_flame.is_flame]
    if existing:
        print(f"[flame/embed] scene already holds flames {existing}", flush=True)
        return
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.thyllore.flame_add()
    print(f"[flame/embed] added {bpy.context.active_object.name}", flush=True)


def find_view3d_region():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for region in area.regions:
                if region.type == "WINDOW":
                    return window, area, region
    return None


def take_screenshot():
    path = str(Path(SCREENSHOT_DIR) / "flame_viewport.png")
    found = find_view3d_region()
    if found is None:
        print("[flame/embed] no VIEW_3D area for screenshot", flush=True)
    else:
        window, area, region = found
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.screen.screenshot_area(filepath=path)
        print(f"[flame/embed] screenshot -> {path}", flush=True)
    if QUIT_AFTER_SCREENSHOT:
        bpy.ops.wm.quit_blender()
    return None


add_campfire()
if SCREENSHOT_DIR:
    bpy.app.timers.register(take_screenshot, first_interval=SCREENSHOT_DELAY_SECONDS)
