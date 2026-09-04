import os
from pathlib import Path

import bpy

SCREENSHOT_DIR = os.environ.get("THYLLORE_SCREENSHOT_DIR", "")
SCREENSHOT_DELAY_SECONDS = float(os.environ.get("THYLLORE_SCREENSHOT_DELAY", "3"))
QUIT_AFTER_SCREENSHOT = os.environ.get("THYLLORE_SCREENSHOT_QUIT", "") == "1"


def add_water():
    existing = [o.name for o in bpy.context.scene.objects if o.thyllore_water.is_water]
    if existing:
        print(f"[water/embed] scene already holds water {existing}", flush=True)
        return
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.thyllore.water_add()
    print(f"[water/embed] added {bpy.context.active_object.name}", flush=True)


def find_view3d_region():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for region in area.regions:
                if region.type == "WINDOW":
                    return window, area, region
    return None


def orbit_view_once():
    found = find_view3d_region()
    if found is None:
        return None
    window, area, region = found
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.view3d.view_orbit(angle=0.6, type="ORBITLEFT")
    print("[water/embed] orbited view to check stale-frame clearing", flush=True)
    return None


def take_screenshot():
    path = str(Path(SCREENSHOT_DIR) / "water_viewport.png")
    found = find_view3d_region()
    if found is None:
        print("[water/embed] no VIEW_3D area for screenshot", flush=True)
    else:
        window, area, region = found
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.screen.screenshot_area(filepath=path)
        print(f"[water/embed] screenshot -> {path}", flush=True)
    if QUIT_AFTER_SCREENSHOT:
        bpy.ops.wm.quit_blender()
    return None


add_water()
if SCREENSHOT_DIR:
    bpy.app.timers.register(orbit_view_once, first_interval=max(SCREENSHOT_DELAY_SECONDS - 2.0, 0.5))
    bpy.app.timers.register(take_screenshot, first_interval=SCREENSHOT_DELAY_SECONDS)
