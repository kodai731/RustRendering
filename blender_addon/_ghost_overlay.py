"""Ghost forecast overlay for the Graph Editor.

Curve Copilot is a preview feature: it shows the predicted curve as a
non-destructive ghost line so the animator can judge it, without touching the
real FCurve. This module owns a GPU draw handler on the Graph Editor that
renders the currently stored forecast in curve space.

The stored points are ``(frame, value)`` in FCurve coordinates; they are mapped
to pixels each redraw via the region's ``view2d``, so the ghost tracks pan/zoom.
"""
from __future__ import annotations

from typing import List, Tuple

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

_ghost_points: List[Tuple[float, float]] = []
_draw_handle = None

_GHOST_LINE_COLOR = (1.0, 0.55, 0.1, 0.95)
_GHOST_POINT_COLOR = (1.0, 0.75, 0.2, 1.0)
_GHOST_LINE_WIDTH = 2.0
_GHOST_POINT_SIZE = 6.0


def set_ghost(points: List[Tuple[float, float]]) -> None:
    global _ghost_points
    _ghost_points = list(points)
    _tag_graph_editors_redraw()


def clear_ghost() -> None:
    global _ghost_points
    _ghost_points = []
    _tag_graph_editors_redraw()


def has_ghost() -> bool:
    return bool(_ghost_points)


def _tag_graph_editors_redraw() -> None:
    window_manager = bpy.context.window_manager
    if window_manager is None:
        return
    for window in window_manager.windows:
        for area in window.screen.areas:
            if area.type == "GRAPH_EDITOR":
                area.tag_redraw()


def _draw_callback() -> None:
    if not _ghost_points:
        return
    region = bpy.context.region
    if region is None:
        return

    view_to_region = region.view2d.view_to_region
    coords = [view_to_region(frame, value, clip=False) for frame, value in _ghost_points]

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(_GHOST_LINE_WIDTH)

    shader.bind()
    shader.uniform_float("color", _GHOST_LINE_COLOR)
    batch_for_shader(shader, "LINE_STRIP", {"pos": coords}).draw(shader)

    gpu.state.point_size_set(_GHOST_POINT_SIZE)
    shader.uniform_float("color", _GHOST_POINT_COLOR)
    batch_for_shader(shader, "POINTS", {"pos": coords}).draw(shader)

    gpu.state.line_width_set(1.0)
    gpu.state.blend_set("NONE")


def register() -> None:
    global _draw_handle
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceGraphEditor.draw_handler_add(
            _draw_callback, (), "WINDOW", "POST_PIXEL"
        )


def unregister() -> None:
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceGraphEditor.draw_handler_remove(_draw_handle, "WINDOW")
        _draw_handle = None
    clear_ghost()
