"""Roboflow CSV export to YOLO txt labels. Every row becomes class 0 (person).

Expects the Roboflow "CSV" export format: a header
    filename,width,height,class,xmin,ymin,xmax,ymax
with one row per box (boxes in pixel coordinates, multiple rows per filename).

Writes labels only. The (large) images stay where they are, and tile_dataset.py
reads them from there. Written for the HERIDAL Roboflow re-export, which ships
boxes as a CSV instead of VOC xml.

    python data_prep/csv_to_yolo.py --src datasets/raw/heridal/heridal/train/_annotations.csv \
        --dst datasets/heridal_labels/train
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="the _annotations.csv file")
    ap.add_argument("--images", type=Path, help="folder with the images (default: same folder as --src)")
    ap.add_argument("--dst", type=Path, required=True, help="folder to write the .txt labels into")
    ap.add_argument("--names", nargs="*", help="keep only these class names (default: all)")
    a = ap.parse_args()

    images_root = a.images or a.src.parent
    index = {p.stem: p for p in images_root.rglob("*") if p.suffix.lower() in IMG_EXT}
    a.dst.mkdir(parents=True, exist_ok=True)

    rows_by_file = defaultdict(list)
    with a.src.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("filename"):
                rows_by_file[row["filename"]].append(row)

    n_files = n_boxes = n_skipped = n_no_image = 0
    for filename, rows in sorted(rows_by_file.items()):
        stem = Path(filename).stem
        img = index.get(stem)
        if img is not None:
            with Image.open(img) as im:
                w, h = im.size  # trust the real image over the csv
        else:
            n_no_image += 1
            w = int(float(rows[0].get("width", 0)))
            h = int(float(rows[0].get("height", 0)))
        if w <= 0 or h <= 0:
            print(f"skip {filename}: no image found and no width/height in the csv")
            continue

        lines = []
        for row in rows:
            if a.names and row.get("class", "").strip() not in a.names:
                continue
            x1, y1, x2, y2 = (float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
            x1, y1, x2, y2 = max(x1, 0.0), max(y1, 0.0), min(x2, w), min(y2, h)
            if x2 - x1 < 1 or y2 - y1 < 1:
                n_skipped += 1
                continue
            lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (a.dst / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_files += 1
        n_boxes += len(lines)

    print(f"{n_files} label files written to {a.dst}, {n_boxes} boxes, {n_skipped} degenerate boxes skipped")
    if n_no_image:
        print(f"warning: {n_no_image} rows had no matching image; the csv width/height was used instead")


if __name__ == "__main__":
    main()
