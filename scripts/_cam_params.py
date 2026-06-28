import bpy, sys
from mathutils import Vector

argv = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
if argv:
    if argv[0].endswith('.blend'):
        bpy.ops.wm.open_mainfile(filepath=argv[0])
    else:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        bpy.ops.wm.usd_import(filepath=argv[0])

arms=[o for o in bpy.data.objects if o.type=='ARMATURE']
arm=max(arms, key=lambda a: len(a.data.bones))
bpy.context.scene.frame_set(1)
lo=Vector((1e18,)*3); hi=Vector((-1e18,)*3)
w=arm.matrix_world
for pb in arm.pose.bones:
    for p in (pb.head, pb.tail):
        wp=w@p
        for i in range(3):
            lo[i]=min(lo[i],wp[i]); hi[i]=max(hi[i],wp[i])
center=(lo+hi)*0.5
size=max((hi-lo).x,(hi-lo).y,(hi-lo).z,0.5)
dist=size*2.0
pos=center+Vector((0,-dist,0))
print("CENTER %.4f %.4f %.4f" % (center.x,center.y,center.z))
print("SIZE %.4f DIST %.4f" % (size,dist))
print("CAMPOS %.4f %.4f %.4f" % (pos.x,pos.y,pos.z))
print("ENGINE_ARGS --camera-pos %.3f,%.3f,%.3f --camera-target %.3f,%.3f,%.3f --fov 39.6" % (pos.x,pos.y,pos.z,center.x,center.y,center.z))
