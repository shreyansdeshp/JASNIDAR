"""Tile VisDrone (official train and val splits) to 640 px, then run a fast sanity check.

    python data_prep/run_visdrone_tiling.py              # tile both splits, then check
    python data_prep/run_visdrone_tiling.py --check-only # only check an existing output

Input is the converted dataset (datasets/visdrone), output is datasets/visdrone_tiled.
The check only counts files and reads label text, it does not open images, so it takes seconds.
For the slower full check (box sizes, preview images) run:
    python data_prep/check_dataset.py datasets/visdrone_tiled --preview 20
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tile(src: Path, dst: Path, split: str, tile_px: int, overlap: float):
    cmd = [
        sys.executable, "-I", str(ROOT / "data_prep" / "tile_dataset.py"),
        "--images", str(src / "images" / split),
        "--labels", str(src / "labels" / split),
        "--dst", str(dst),
        "--force-split", split,  # keep the official split, do not re-hash
        "--tile", str(tile_px), "--overlap", str(overlap), "--scale", "1.0",
    ]
    print(f"\n== tiling {split} ==", flush=True)
    subprocess.run(cmd, check=True)


def check(dst: Path) -> bool:
    ok = True
    for split in ("train", "val"):
        imgs = {p.stem for p in (dst / "images" / split).glob("*.jpg")}
        lbls = {p.stem for p in (dst / "labels" / split).glob("*.txt")}
        boxes = bad = 0
        for p in (dst / "labels" / split).glob("*.txt"):
            for line in p.read_text().splitlines():
                v = line.split()
                if not v:
                    continue
                boxes += 1
                if len(v) != 5 or v[0] != "0" or not all(0 <= float(x) <= 1 for x in v[1:]):
                    bad += 1
        # a tile's source is the name before _x<left>_y<top>
        src_train = {s.rsplit("_x", 1)[0] for s in imgs} if split == "train" else set()
        print(f"[{split}] {len(imgs)} images, {len(lbls)} labels, {boxes} boxes, {bad} bad lines, "
              f"{len(imgs - lbls)} missing labels, {len(lbls - imgs)} labels without image")
        ok &= not bad and imgs == lbls
        if split == "train":
            train_src = src_train
        else:
            val_src = {s.rsplit("_x", 1)[0] for s in imgs}
    shared = train_src & val_src
    print(f"source images in both train and val: {len(shared)}")
    return ok and not shared


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "datasets" / "visdrone")
    ap.add_argument("--dst", type=Path, default=ROOT / "datasets" / "visdrone_tiled")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()

    if not a.check_only:
        for split in ("val", "train"):
            tile(a.src, a.dst, split, a.tile, a.overlap)
    print("\n== check ==")
    print("OK" if check(a.dst) else "PROBLEMS FOUND")


if __name__ == "__main__":
    main()
