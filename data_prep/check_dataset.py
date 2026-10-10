"""Sanity-check a YOLO dataset folder before training.

Checks every label file (format, class, coordinate range), counts images per split,
reports box sizes in pixels, and warns if the same source image appears in two splits.
With --preview N it draws the boxes on N random images per split, so you can look at them.

    python data_prep/check_dataset.py datasets/heridal --preview 20 --imgsz 640

Open the preview images. Shifted, flipped or missing boxes are the most common reason
a training run quietly gives bad results.
"""
import argparse
import random
import statistics
from pathlib import Path

from PIL import Image, ImageDraw

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def source_of(stem: str) -> str:
    """Tile names look like <source>_x<left>_y<top>; anything else is its own source."""
    return stem.rsplit("_x", 1)[0] if "_x" in stem and "_y" in stem else stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset folder containing images/<split> and labels/<split>")
    ap.add_argument("--imgsz", type=int, default=640, help="training image size, to judge box sizes")
    ap.add_argument("--preview", type=int, default=0, help="draw boxes on N random images per split")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    splits = sorted(d.name for d in (a.root / "images").iterdir() if d.is_dir())
    if not splits:
        raise SystemExit(f"No split folders under {a.root / 'images'}")

    problems = 0
    sources = {}
    for split in splits:
        img_dir, lbl_dir = a.root / "images" / split, a.root / "labels" / split
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        n_empty = n_boxes = missing = 0
        heights_px = []
        for img in imgs:
            sources.setdefault(source_of(img.stem), set()).add(split)
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                missing += 1
                continue
            with Image.open(img) as im:
                w, h = im.size
            scale = a.imgsz / max(w, h)
            lines = [x for x in lbl.read_text().splitlines() if x.strip()]
            if not lines:
                n_empty += 1
            for line in lines:
                p = line.split()
                try:
                    ok = len(p) == 5 and p[0] == "0" and all(0.0 <= float(v) <= 1.0 for v in p[1:])
                except ValueError:
                    ok = False
                if not ok:
                    problems += 1
                    print(f"  bad line in {lbl.name}: {line!r}")
                    continue
                n_boxes += 1
                heights_px.append(float(p[4]) * h * scale)
        extra = len({p.stem for p in lbl_dir.glob("*.txt")} - {i.stem for i in imgs})
        print(f"[{split}] {len(imgs)} images, {n_empty} empty, {n_boxes} boxes, "
              f"{missing} missing labels, {extra} labels without an image")
        problems += missing + extra
        if heights_px:
            hp = sorted(heights_px)
            small = sum(v < 16 for v in hp) / len(hp)
            print(f"   box height at imgsz {a.imgsz}: min {hp[0]:.0f}px, median {statistics.median(hp):.0f}px, "
                  f"90th pct {hp[int(0.9 * (len(hp) - 1))]:.0f}px, under 16px: {small:.0%}")

    leaks = {s: v for s, v in sources.items() if len(v) > 1}
    if leaks:
        problems += len(leaks)
        print(f"WARNING: {len(leaks)} source images appear in more than one split, e.g. {next(iter(leaks))}")

    if a.preview:
        out = a.root / "_preview"
        out.mkdir(exist_ok=True)
        rng = random.Random(a.seed)
        for split in splits:
            imgs = sorted(p for p in (a.root / "images" / split).iterdir() if p.suffix.lower() in IMG_EXT)
            for img in rng.sample(imgs, min(a.preview, len(imgs))):
                with Image.open(img) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    d = ImageDraw.Draw(im)
                    lbl = a.root / "labels" / split / f"{img.stem}.txt"
                    for line in lbl.read_text().splitlines() if lbl.exists() else []:
                        p = line.split()
                        if len(p) == 5:
                            cx, cy, bw, bh = (float(v) for v in p[1:])
                            d.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                                        outline=(255, 0, 0), width=max(2, w // 300))
                    im.thumbnail((1280, 1280))
                    im.save(out / f"{split}_{img.stem}.jpg", quality=90)
        print(f"Previews written to {out}. Open a few and check the boxes sit on people.")

    print("OK" if not problems else f"{problems} problem(s) found, see above")


if __name__ == "__main__":
    main()
