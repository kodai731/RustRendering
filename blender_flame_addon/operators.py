from __future__ import annotations

import os

import bpy


class THYLLORE_OT_flame_add(bpy.types.Operator):

    bl_idname = "thyllore.flame_add"
    bl_label = "Add Flame"

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        cursor_location = context.scene.cursor.location

        obj = bpy.data.objects.new("Flame", None)
        obj.empty_display_type = "CUBE"
        obj.empty_display_size = 0.5
        obj.location = cursor_location

        context.collection.objects.link(obj)

        obj.thyllore_flame.is_flame = True
        obj.thyllore_flame.preset = "campfire"

        context.view_layer.objects.active = obj
        obj.select_set(True)

        return {"FINISHED"}


class THYLLORE_OT_flame_render_sequence(bpy.types.Operator):
    bl_idname = "thyllore.flame_render_sequence"
    bl_label = "Render Flame Sequence"

    out_dir: bpy.props.StringProperty(subtype="DIR_PATH", default="//flame/")
    use_scene_depth: bpy.props.BoolProperty(default=True)

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        from .render import render_flame_sequence

        obj = context.view_layer.objects.active
        if obj is None or not obj.thyllore_flame.is_flame:
            self.report({"ERROR"}, "No active flame object")
            return {"CANCELLED"}

        out_dir = os.path.abspath(self.out_dir)
        os.makedirs(out_dir, exist_ok=True)

        scene = context.scene
        paths = render_flame_sequence(
            scene, obj, out_dir, int(scene.frame_start), int(scene.frame_end), write_npy=False, use_scene_depth=self.use_scene_depth
        )
        self.report({"INFO"}, f"wrote {len(paths)} frames")
        return {"FINISHED"}


class THYLLORE_OT_flame_setup_compositor(bpy.types.Operator):
    bl_idname = "thyllore.flame_setup_compositor"
    bl_label = "Setup Flame Compositor"

    out_dir: bpy.props.StringProperty(subtype="DIR_PATH", default="//flame/")

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        from .compositor import setup_flame_compositor

        obj = context.view_layer.objects.active
        if obj is None or not obj.thyllore_flame.is_flame:
            self.report({"ERROR"}, "No active flame object")
            return {"CANCELLED"}

        out_dir = os.path.abspath(self.out_dir)
        os.makedirs(out_dir, exist_ok=True)

        scene = context.scene
        setup_flame_compositor(scene, obj, out_dir, int(scene.frame_start), int(scene.frame_end))
        self.report({"INFO"}, "Compositor nodes set up")
        return {"FINISHED"}


class THYLLORE_OT_flame_dump_viewport(bpy.types.Operator):
    bl_idname = "thyllore.flame_dump_viewport"
    bl_label = "Dump Flame Viewport Texture"

    out_dir: bpy.props.StringProperty(subtype="DIR_PATH", default="/screenshots/")

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D"

    def execute(self, context):
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        from mathutils import Vector

        from . import draw_handler

        renderer = next(iter(draw_handler._renderers.values()), None)
        if renderer is None or renderer.color is None:
            self.report({"ERROR"}, "no flame has been drawn yet")
            return {"CANCELLED"}

        width, height = renderer._w, renderer._h
        rows = renderer.color.read().to_list()
        region = context.region
        origin = location_3d_to_region_2d(region, context.region_data, Vector((0.0, 0.0, 0.0)))
        origin_y = origin.y if origin is not None else float("-inf")
        below = above = 0
        alpha_rows = []
        for y, row in enumerate(rows):
            count = sum(1 for px in row if px[3] > 0.0)
            if count:
                alpha_rows.append(y)
                if y < origin_y:
                    below += count
                else:
                    above += count

        out_dir = self.out_dir if os.path.isdir(self.out_dir) else os.path.expanduser("~")
        image = bpy.data.images.new("thyllore_flame_dump", width, height, alpha=True, float_buffer=True)
        image.pixels.foreach_set([min(1.0, c * 4.0) if i % 4 != 3 else 1.0 for row in rows for px in row for i, c in enumerate(px)])
        image.filepath_raw = os.path.join(out_dir, "flame_viewport_dump.png")
        image.file_format = "PNG"
        image.save()
        bpy.data.images.remove(image)

        summary = (
            f"size={width}x{height} region={region.width}x{region.height} origin_y={origin_y:.0f} "
            f"alpha_rows={alpha_rows[0] if alpha_rows else None}-{alpha_rows[-1] if alpha_rows else None} "
            f"below_origin={below} above={above} -> {os.path.join(out_dir, 'flame_viewport_dump.png')}"
        )
        print(f"[Thyllore Flame] dump: {summary}", flush=True)
        self.report({"INFO"}, summary)
        return {"FINISHED"}
