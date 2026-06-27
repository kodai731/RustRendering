import bpy, numpy as np, os, sys

argv = sys.argv[sys.argv.index('--')+1:]
left_path, right_path, out_path = argv[0], argv[1], argv[2]

def load_rgb_on_white(path):
    img = bpy.data.images.load(os.path.abspath(path))
    w, h = img.size
    buf = np.empty(w*h*4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    a = buf.reshape(h, w, 4)
    a = np.flipud(a)  # bpy is bottom-up
    rgb = a[..., :3]
    alpha = a[..., 3:4]
    out = rgb*alpha + (1.0-alpha)*1.0  # composite over white
    bpy.data.images.remove(img)
    return out  # (h,w,3) float 0..1

left = load_rgb_on_white(left_path)
right = load_rgb_on_white(right_path)

h = max(left.shape[0], right.shape[0])
def pad(a):
    ph = h - a.shape[0]
    if ph: a = np.pad(a, ((0,ph),(0,0),(0,0)), constant_values=1.0)
    return a
left, right = pad(left), pad(right)

gap = 8
sep = np.ones((h, gap, 3), dtype=np.float32)
canvas = np.concatenate([left, sep, right], axis=1)
H, W = canvas.shape[:2]

# save via a new bpy image
out_img = bpy.data.images.new("compare", width=W, height=H, alpha=False)
rgba = np.concatenate([canvas, np.ones((H,W,1),dtype=np.float32)], axis=2)
rgba = np.flipud(rgba)  # back to bottom-up for bpy
out_img.pixels.foreach_set(rgba.reshape(-1))
out_img.filepath_raw = os.path.abspath(out_path)
out_img.file_format = 'PNG'
out_img.save()
print("COMPOSED:", out_path, (W, H))
