"""Cut large images (and their YOLO labels) into overlapping square tiles.

Why: a 4000x3000 photo shrunk to 640 px turns a person into a few pixels.
Tiles keep the original detail. Optionally upscale first (--scale) when the
source was shot higher than your flight altitude, so people look the size they
will at your altitude.

Details
- Tiles overlap (--overlap), so a person cut by one tile edge is whole in the next.
- A box is kept in a tile only if at least --min-vis of it is inside the tile
  (and the visible part is at least --min-px wide and tall). Kept boxes are clipped.
- Tiles without people are mostly dropped: about --empty-ratio empty tiles are
  kept per tile with people.
- Train/val is decided per SOURCE image (a hash of its name), never per tile, so
  tiles of one photo never end up in both splits. Use --force-split to put
  everything in one split (for example the official test images).
- Tile names are <prefix><source>_x<left>_y<top>.jpg, so the source is recoverable.

Labels are read as YOLO txt (class is ignored, everything becomes class 0):

    python data_prep/tile_dataset.py --images datasets/raw/heridal/trainImages \
        --labels datasets/heridal_labels/train --dst datasets/heridal \
        --scale 1.5 --tile 640 --overlap 0.2 --val-frac 0.1
"""
import argparse
import hashlib
import random
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # local, trusted data

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def positions(length: int, tile: int, stride: int):
    """Left/top offsets that cover [0, length), with the last tile flush to the edge."""
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] + tile < length:
        pos.append(length - tile)
    return pos


def split_of(stem: str, val_frac: float, seed: int) -> str:
    h = int(hashlib.md5(f"{seed}-{stem}".encode()).hexdigest(), 16) % 10000
    return "val" if h < val_frac * 10000 else "train"


def read_boxes(label_path: Path, w: int, h: int):
    """YOLO txt -> list of (x1, y1, x2, y2) in pixels of a w x h image."""
    boxes = []
    if label_path is not None and label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            cx, cy, bw, bh = (float(v) for v in p[1:])
            boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def make_tiles(w, h, boxes, tile, overlap, min_vis, min_px):
    """Return (tiles, n_lost). Each tile is (x0, y0, x1, y1, yolo_lines)."""
    stride = max(1, int(tile * (1 - overlap)))
    kept = [False] * len(boxes)
    tiles = []
    for y0 in positions(h, tile, stride):
        for x0 in positions(w, tile, stride):
            x1, y1 = min(x0 + tile, w), min(y0 + tile, h)
            tw, th = x1 - x0, y1 - y0
            lines = []
            for i, (bx1, by1, bx2, by2) in enumerate(boxes):
                area = (bx2 - bx1) * (by2 - by1)
                if area <= 0:
                    continue
                ix1, iy1, ix2, iy2 = max(bx1, x0), max(by1, y0), min(bx2, x1), min(by2, y1)
                iw, ih = ix2 - ix1, iy2 - iy1
                if iw <= 0 or ih <= 0 or iw * ih / area < min_vis or iw < min_px or ih < min_px:
                    continue
                kept[i] = True
                lines.append(
                    f"0 {((ix1 + ix2) / 2 - x0) / tw:.6f} {((iy1 + iy2) / 2 - y0) / th:.6f} {iw / tw:.6f} {ih / th:.6f}"
                )
            tiles.append((x0, y0, x1, y1, lines))
    return tiles, kept.count(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="folder with the large images (searched recursively)")
    ap.add_argument("--labels", type=Path, required=True, help="folder with YOLO .txt labels, same file stems")
    ap.add_argument("--dst", type=Path, required=True, help="output dataset folder")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--scale", type=float, default=1.0, help="resize the image by this factor before tiling")
    ap.add_argument("--min-vis", type=float, default=0.4, help="min fraction of a box that must lie inside a tile")
    ap.add_argument("--min-px", type=float, default=4.0, help="min visible box width and height in tile pixels")
    ap.add_argument("--empty-ratio", type=float, default=0.1, help="empty tiles kept per tile with people")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--force-split", choices=["train", "val", "test"], help="put every tile in this split")
    ap.add_argument("--prefix", default="", help="added to every output file name, e.g. heridal_")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    images = sorted(p for p in a.images.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not images:
        raise SystemExit(f"No images found under {a.images}")

    totals = {}  # split -> [positive tiles, empty tiles, boxes]
    n_lost = n_boxes_in = n_no_label = 0
    for img_path in images:
        label_path = a.labels / f"{img_path.stem}.txt"
        if not label_path.exists():
            n_no_label += 1  # treated as a source image with no people
        split = a.force_split or split_of(img_path.stem, a.val_frac, a.seed)

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            if a.scale != 1.0:
                im = im.resize((round(im.width * a.scale), round(im.height * a.scale)), Image.LANCZOS)
            w, h = im.size
            boxes = read_boxes(label_path, w, h)
            tiles, lost = make_tiles(w, h, boxes, a.tile, a.overlap, a.min_vis, a.min_px)

            pos = [t for t in tiles if t[4]]
            neg = [t for t in tiles if not t[4]]
            rng = random.Random(f"{a.seed}-{img_path.stem}")
            want = a.empty_ratio * len(pos)
            n_neg = min(len(neg), int(want) + (rng.random() < want - int(want)))
            chosen = pos + rng.sample(neg, n_neg)

            out_img = a.dst / "images" / split
            out_lbl = a.dst / "labels" / split
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)
            for x0, y0, x1, y1, lines in chosen:
                name = f"{a.prefix}{img_path.stem}_x{x0}_y{y0}"
                im.crop((x0, y0, x1, y1)).save(out_img / f"{name}.jpg", quality=95)
                (out_lbl / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        t = totals.setdefault(split, [0, 0, 0])
        t[0] += len(pos)
        t[1] += n_neg
        t[2] += sum(len(x[4]) for x in pos)
        n_lost += lost
        n_boxes_in += len(boxes)

    print(f"{len(images)} source images, {n_boxes_in} boxes in")
    for split, (npos, nneg, nbox) in sorted(totals.items()):
        print(f"  {split}: {npos} tiles with people, {nneg} empty tiles, {nbox} boxes")
    if n_no_label:
        print(f"  note: {n_no_label} images had no label file (treated as having no people)")
    if n_lost:
        print(f"  warning: {n_lost} boxes were not kept in any tile (too small or always cut); "
              f"try a higher --overlap or a lower --min-vis")
    print(f"  output: {a.dst}")


if __name__ == "__main__":
    main()
