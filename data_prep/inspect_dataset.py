"""First look at a downloaded dataset: file types, image sizes and one sample label.

Run this before converting anything, so you know which format you actually have.

    python data_prep/inspect_dataset.py datasets/raw/heridal
"""
import argparse
import collections
import random
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SKIP_NAMES = {"readme.txt", "license.txt", "requirements.txt"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--n", type=int, default=30, help="how many images to measure")
    a = ap.parse_args()

    files = [p for p in a.root.rglob("*") if p.is_file()]
    print(f"{len(files)} files under {a.root}")
    print("file types:")
    for ext, c in collections.Counter(p.suffix.lower() for p in files).most_common():
        print(f"  {ext or '(none)'}: {c}")

    print("folders with the most files:")
    folders = collections.Counter(p.parent.relative_to(a.root).as_posix() for p in files)
    for d, c in folders.most_common(10):
        print(f"  {d}: {c}")

    imgs = [p for p in files if p.suffix.lower() in IMG_EXT]
    random.Random(0).shuffle(imgs)
    sizes = collections.Counter()
    for p in imgs[: a.n]:
        with Image.open(p) as im:
            sizes[im.size] += 1
    print(f"image sizes (sample of {min(a.n, len(imgs))}):")
    for (w, h), c in sizes.most_common():
        print(f"  {w}x{h}: {c}")

    for ext in (".txt", ".xml", ".json", ".csv"):
        cand = sorted(p for p in files if p.suffix.lower() == ext and p.name.lower() not in SKIP_NAMES)
        if cand:
            print(f"\nfirst {ext} file: {cand[0]}")
            print(cand[0].read_text(errors="replace")[:600])


if __name__ == "__main__":
    main()
