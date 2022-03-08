"""
Use this script in case you saved rehearsal memory on a computer A, but then want
to resume training, using those rehearsal samples, on a computer B.

Because for ImageNet we save the path, which may be different on each computer.
"""

import sys
import glob
import os
import shutil

import numpy as np

memory_path = sys.argv[1]
new_base_path = sys.argv[2]

if os.path.isdir(memory_path):
    memory_paths = glob.glob(os.path.abspath(os.path.join(memory_path, "memory_*.npz")))
else:
    memory_paths = [memory_path]

print(memory_paths)

for p in sorted(memory_paths):
    psrc = p
    if not os.path.exists(f"{p}_original"):
        shutil.copy(p, f"{p}_original")
    else:
        psrc = f"{p}_original"
    print(p)

    data = np.load(p)
    x = []
    for img_path in data["x"]:
        id_ = str(img_path).lstrip("b'").rstrip("'").split("train")[-1][1:]

        x.append(os.path.join(new_base_path, "train", id_))

    np.savez(
        p,
        x=np.array(x), y=data["y"], t=data["t"]
    )
    print("Done!")
