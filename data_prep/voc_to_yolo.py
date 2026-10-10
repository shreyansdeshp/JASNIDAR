"""PASCAL VOC xml to YOLO txt labels. Every object becomes class 0 (person).

Writes labels only. The (large) images stay where they are, and tile_dataset.py
reads them from there. Written for HERIDAL, whose Zenodo release is in VOC format.
If your download is not xml, run inspect_dataset.py first and adapt this.

    python data_prep/voc_to_yolo.py --src datasets/raw/heridal/trainImages \
        --dst datasets/heridal_labels/train
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="folder with .xml files (searched recursively)")
    ap.add_argument("--images", type=Path, help="folder with the images (default: same as --src)")
    ap.add_argument("--dst", type=Path, required=True, help="folder to write the .txt labels into")
    ap.add_argument("--names", nargs="*", help="keep only objects with these names (default: all)")
    a = ap.parse_args()

    images_root = a.images or a.src
    index = {p.stem: p for p in images_root.rglob("*") if p.suffix.lower() in IMG_EXT}
    a.dst.mkdir(parents=True, exist_ok=True)

    n_files = n_boxes = n_skipped = n_no_image = 0
    for xml_path in sorted(a.src.rglob("*.xml")):
        root = ET.parse(xml_path).getroot()
        stem = xml_path.stem
        img = index.get(stem)
        if img is not None:
            with Image.open(img) as im:
                w, h = im.size  # trust the real image over the xml
        else:
            n_no_image += 1
            w = int(float(root.findtext("size/width", "0")))
            h = int(float(root.findtext("size/height", "0")))
        if w <= 0 or h <= 0:
            print(f"skip {xml_path.name}: no image found and no size in the xml")
            continue

        lines = []
        for obj in root.iter("object"):
            if a.names and obj.findtext("name", "").strip() not in a.names:
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            x1, y1, x2, y2 = (float(bb.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
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
        print(f"warning: {n_no_image} xml files had no matching image; the xml size was used instead")


if __name__ == "__main__":
    main()
