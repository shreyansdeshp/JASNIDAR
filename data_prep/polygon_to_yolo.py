"""YOLO segmentation polygons to YOLO detection boxes. Every object becomes class 0 (person).

Each input line is `class x1 y1 x2 y2 ...` (normalised). The output line is the
bounding box of those points: `0 cx cy w h`. Images are copied next to the labels.
Written for the Roboflow UAV-SAR-Human-Detection test set, which exports polygons.

    python data_prep/polygon_to_yolo.py \
        --src ../UAV-SAR-Human-Detection-Dataset.v1-v1.yolov11/test \
        --dst datasets/uavsar_test
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="split folder containing images/ and labels/")
    ap.add_argument("--dst", type=Path, required=True, help="output folder (gets images/ and labels/)")
    a = ap.parse_args()

    (a.dst / "images").mkdir(parents=True, exist_ok=True)
    (a.dst / "labels").mkdir(parents=True, exist_ok=True)

    n_files = n_boxes = n_skipped = 0
    for txt in sorted((a.src / "labels").glob("*.txt")):
        lines = []
        for raw in txt.read_text().splitlines():
            f = raw.split()
            if len(f) == 5:  # already a box
                lines.append("0 " + " ".join(f[1:]))
                continue
            if len(f) < 7 or len(f) % 2 == 0:  # need class + at least 3 (x, y) pairs
                n_skipped += 1
                continue
            xs = [float(v) for v in f[1::2]]
            ys = [float(v) for v in f[2::2]]
            x1, y1, x2, y2 = max(min(xs), 0.0), max(min(ys), 0.0), min(max(xs), 1.0), min(max(ys), 1.0)
            if x2 <= x1 or y2 <= y1:
                n_skipped += 1
                continue
            lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
        (a.dst / "labels" / txt.name).write_text("\n".join(lines) + ("\n" if lines else ""))
        n_files += 1
        n_boxes += len(lines)

        img = next((p for p in (a.src / "images").glob(txt.stem + ".*")), None)
        if img is not None:
            shutil.copy2(img, a.dst / "images" / img.name)
        else:
            print(f"warning: no image for {txt.name}")

    print(f"{n_files} label files written to {a.dst / 'labels'}, {n_boxes} boxes, {n_skipped} degenerate objects skipped")


if __name__ == "__main__":
    main()
