"""VisDrone-DET to YOLO format, keeping only people (pedestrian + people -> class 0).

Expects the official folder layout:  <src>/images/*.jpg  and  <src>/annotations/*.txt
Each annotation line is:  x,y,w,h,score,category,truncation,occlusion
Categories: 0 ignored region, 1 pedestrian, 2 people, 3 bicycle, 4 car, ...
Lines with score 0 (ignored) and every category except 1 and 2 are dropped.

Images without people are kept as negatives, but only a few of them
(--empty-ratio empty images per image that has people), so cars and buildings
do not dominate the data.

    python data_prep/convert_visdrone.py --src datasets/raw/visdrone/VisDrone2019-DET-train \
        --dst datasets/visdrone --split train
"""
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

KEEP = {1, 2}


def convert_annotation(path: Path, w: int, h: int):
    """Return (yolo_lines, dropped_count) for one annotation file."""
    lines, dropped = [], 0
    for raw in path.read_text().splitlines():
        row = raw.strip().split(",")
        if len(row) < 6:
            continue
        x, y, bw, bh = (float(v) for v in row[:4])
        score, cat = int(float(row[4])), int(float(row[5]))
        if score == 0 or cat not in KEEP:
            continue
        x1, y1, x2, y2 = max(x, 0.0), max(y, 0.0), min(x + bw, w), min(y + bh, h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            dropped += 1
            continue
        lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="e.g. VisDrone2019-DET-train")
    ap.add_argument("--dst", type=Path, required=True, help="output dataset folder")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--empty-ratio", type=float, default=0.1,
                    help="empty images to keep per image with people (default 0.1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--symlink", action="store_true", help="link images instead of copying")
    a = ap.parse_args()

    img_dir, ann_dir = a.src / "images", a.src / "annotations"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise SystemExit(f"Expected {img_dir} and {ann_dir}. Run inspect_dataset.py to see the real layout.")

    positives, empties, n_boxes, n_dropped = [], [], 0, 0
    for img in sorted(img_dir.glob("*.jpg")):
        ann = ann_dir / f"{img.stem}.txt"
        if not ann.exists():
            continue
        with Image.open(img) as im:
            w, h = im.size
        lines, dropped = convert_annotation(ann, w, h)
        n_dropped += dropped
        (positives if lines else empties).append((img, lines))
        n_boxes += len(lines)

    rng = random.Random(a.seed)
    rng.shuffle(empties)
    keep_empty = empties[: round(a.empty_ratio * len(positives))]
    kept = positives + keep_empty

    out_img = a.dst / "images" / a.split
    out_lbl = a.dst / "labels" / a.split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    for img, lines in kept:
        target = out_img / img.name
        if a.symlink:
            if not target.exists():
                target.symlink_to(img.resolve())
        else:
            shutil.copy2(img, target)
        (out_lbl / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"{a.split}: {len(kept)} images written ({len(positives)} with people, {len(keep_empty)} empty)")
    print(f"  people boxes: {n_boxes}, degenerate boxes dropped: {n_dropped}")
    print(f"  empty images available: {len(empties)}, kept: {len(keep_empty)}")
    print(f"  output: {a.dst}")


if __name__ == "__main__":
    main()
