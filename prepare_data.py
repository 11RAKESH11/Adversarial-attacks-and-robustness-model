"""Optional: download CIFAR-10 from the Hugging Face mirror and cache it as data/cifar10.npz.

Use this when the official torchvision download host (cs.toronto.edu) is slow or blocked.
advrob.get_loaders() automatically uses data/cifar10.npz when it exists.

    pip install pyarrow          # only needed for this script
    python prepare_data.py
"""
import io
import os
import sys
import urllib.request

import numpy as np

BASE = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
FILES = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}


def load_split(name, out_dir):
    import pyarrow.parquet as pq
    from PIL import Image
    path = os.path.join(out_dir, FILES[name])
    if not os.path.exists(path):
        print(f"downloading {FILES[name]} ...")
        urllib.request.urlretrieve(BASE + FILES[name], path)
    table = pq.read_table(path).to_pydict()
    img_col = "img" if "img" in table else "image"
    x = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in table[img_col]])
    y = np.array(table["label"], dtype=np.int64)
    return x, y


def main(out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    xtr, ytr = load_split("train", out_dir)
    xte, yte = load_split("test", out_dir)
    print("train", xtr.shape, ytr.shape, "test", xte.shape, yte.shape, "classes", np.unique(ytr))
    np.savez(os.path.join(out_dir, "cifar10.npz"), xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    print("saved", os.path.join(out_dir, "cifar10.npz"))


if __name__ == "__main__":
    main(*sys.argv[1:])
