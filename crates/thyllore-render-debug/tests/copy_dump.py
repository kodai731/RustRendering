import shutil
import hashlib
import os

src = "log/flame/wall_probe_1786149179.json"
dst = "crates/thyllore-render-debug/tests/data/wall_probe_sample.json"

shutil.copy2(src, dst)

size = os.path.getsize(dst)
with open(dst, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()

print(f"{sha}  {dst}")
print(f"{size} {dst}")
