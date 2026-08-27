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
