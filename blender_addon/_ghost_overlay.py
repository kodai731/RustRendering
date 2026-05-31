"""Ghost forecast overlay for the Graph Editor.

Curve Copilot is a preview feature: it shows the predicted curve(s) as
non-destructive ghost lines so the animator can judge them, without touching the
real FCurves. This module owns a GPU draw handler on the Graph Editor that
renders the currently stored forecasts in curve space.

Each stored curve is a list of ``(frame, value)`` points in FCurve coordinates,
mapped to pixels each redraw via the region's ``view2d`` so the ghost tracks
pan/zoom. When several curves are selected, each gets a distinct palette color.
"""
from __future__ import annotations

from typing import List, Tuple

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

_ghost_curves: List[List[Tuple[float, float]]] = []
_draw_handle = None

# Distinct colors cycled per selected curve (RGBA).
_GHOST_PALETTE = [
    (1.0, 0.55, 0.1, 0.95),   # orange
    (0.2, 0.8, 0.95, 0.95),   # cyan
    (0.45, 0.9, 0.35, 0.95),  # green
    (0.95, 0.35, 0.85, 0.95), # magenta
    (0.95, 0.9, 0.25, 0.95),  # yellow
    (0.95, 0.3, 0.25, 0.95),  # red
    (0.4, 0.55, 1.0, 0.95),   # blue
    (0.95, 0.95, 0.95, 0.95), # white
]
_GHOST_LINE_WIDTH = 2.0
_GHOST_POINT_SIZE = 6.0


def set_ghosts(curves: List[List[Tuple[float, float]]]) -> None:
    """Replace the displayed ghosts with one polyline per selected curve."""
    global _ghost_curves
    _ghost_curves = [list(curve) for curve in curves if curve]
    _tag_graph_editors_redraw()


def clear_ghost() -> None:
    global _ghost_curves
    _ghost_curves = []
    _tag_graph_editors_redraw()


def has_ghost() -> bool:
    return bool(_ghost_curves)


def color_for_index(index: int) -> Tuple[float, float, float, float]:
    """Palette color a given curve index will be drawn with (for logging/UI)."""
    return _GHOST_PALETTE[index % len(_GHOST_PALETTE)]


def _tag_graph_editors_redraw() -> None:
    window_manager = bpy.context.window_manager
    if window_manager is None:
        return
    for window in window_manager.windows:
        for area in window.screen.areas:
            if area.type == "GRAPH_EDITOR":
                area.tag_redraw()


def _draw_callback() -> None:
    if not _ghost_curves:
        return
    region = bpy.context.region
    if region is None:
        return

    view_to_region = region.view2d.view_to_region
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    shader.bind()

    for index, points in enumerate(_ghost_curves):
        color = color_for_index(index)
        coords = [view_to_region(frame, value, clip=False) for frame, value in points]

        gpu.state.line_width_set(_GHOST_LINE_WIDTH)
        shader.uniform_float("color", color)
        batch_for_shader(shader, "LINE_STRIP", {"pos": coords}).draw(shader)

        gpu.state.point_size_set(_GHOST_POINT_SIZE)
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
