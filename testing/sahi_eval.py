"""Score a YOLO model on the fixed test set (datasets/uavsar_test), with SAHI sliced inference.

Every test frame is scored whole: the model runs on overlapping 640 px windows, boxes are
shifted back to frame coordinates, duplicates from the overlaps are merged, and the result
is compared with the original full-frame labels (IoU 0.5). --mode plain skips the slicing
(whole frame squeezed to --imgsz) as a baseline, using the same scoring code.

Setup (run once, in your environment):
    pip install ultralytics sahi
    (SAHI 0.11.20 or newer; with an older SAHI change model_type below to "yolov8")

Usage (from the repo root):
    python testing/sahi_eval.py --weights runs/detect/s1_visdrone/weights/best.pt
    python testing/sahi_eval.py --weights yolo11s.pt --name coco_baseline
    python testing/sahi_eval.py --weights <w> --mode plain

Writes testing/results/<name>_<mode>.json and prints a table. The class filter keeps only
class 0 (person), so COCO weights, which have 80 classes, can be scored too.
"""
import argparse
import json
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png"}
REPORT_CONF = 0.25  # confidence at which precision and recall are quoted


# ---------- scoring (plain python, no extra dependencies) ----------
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(dets, gts, thr):
    """dets: [(img, score, box)], gts: {img: [box]}. Returns [(score, is_true_positive)] sorted by score."""
    used = {img: [False] * len(b) for img, b in gts.items()}
    out = []
    for img, score, box in sorted(dets, key=lambda d: -d[1]):
        best, best_j = thr, -1
        for j, g in enumerate(gts.get(img, [])):
            if used[img][j]:
                continue
            v = iou(box, g)
            if v >= best:
                best, best_j = v, j
        if best_j >= 0:
            used[img][best_j] = True
        out.append((score, best_j >= 0))
    return out


def metrics(dets, gts, thr=0.5):
    n_gt = sum(len(b) for b in gts.values())
    m = match(dets, gts, thr)
    if not n_gt:
        return {"n_gt": 0}
    tp = fp = 0
    prec, rec = [], []
    for _, ok in m:
        tp, fp = tp + ok, fp + (not ok)
        prec.append(tp / (tp + fp))
        rec.append(tp / n_gt)
    # AP: area under the precision envelope (all-point interpolation)
    env = prec[:]
    for i in range(len(env) - 2, -1, -1):
        env[i] = max(env[i], env[i + 1])
    ap, prev_r = 0.0, 0.0
    for p, r in zip(env, rec):
        ap += p * (r - prev_r)
        prev_r = r

    def at(conf):
        k = sum(1 for s, _ in m if s >= conf)
        t = sum(1 for s, ok in m if s >= conf and ok)
        p, r = (t / k if k else 0.0), t / n_gt
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    p, r, f1 = at(REPORT_CONF)
    best = max(((c / 100, *at(c / 100)) for c in range(5, 96, 5)), key=lambda x: x[3])
    return {"n_gt": n_gt, "AP50": ap, f"P@{REPORT_CONF}": p, f"R@{REPORT_CONF}": r, f"F1@{REPORT_CONF}": f1,
            "best_conf": best[0], "best_P": best[1], "best_R": best[2], "best_F1": best[3]}


# ---------- data ----------
def load_gt(data_dir: Path, split: str):
    """Returns {image path: [xyxy boxes in pixels]} from YOLO label files."""
    gts = {}
    for img in sorted((data_dir / "images" / split).iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        w, h = Image.open(img).size
        lbl = data_dir / "labels" / split / f"{img.stem}.txt"
        boxes = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                v = line.split()
                if len(v) == 5:
                    cx, cy, bw, bh = (float(x) for x in v[1:])
                    boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        gts[str(img)] = boxes
    return gts


def condition(path: str) -> str:
    """'test_15m45d_frame29_jpg.rf...' -> '15m45d'; anything else -> 'all'."""
    m = re.search(r"(\d+m\d+d)", Path(path).name)
    return m.group(1) if m else "all"


# ---------- prediction ----------
def predict_all(weights, images, a):
    """Returns [(image, score, xyxy)] for class 0 only."""
    dets = []
    if a.mode == "sahi":
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(weights), confidence_threshold=a.conf, device=a.device)
        for i, img in enumerate(images, 1):
            res = get_sliced_prediction(
                img, model, slice_height=a.slice, slice_width=a.slice,
                overlap_height_ratio=a.overlap, overlap_width_ratio=a.overlap,
                perform_standard_pred=not a.no_full_pass,
                postprocess_type=a.merge, postprocess_match_metric="IOS", postprocess_match_threshold=0.5,
                verbose=0)
            for p in res.object_prediction_list:
                if p.category.id == 0:
                    b = p.bbox
                    dets.append((img, p.score.value, (b.minx, b.miny, b.maxx, b.maxy)))
            if i % 50 == 0:
                print(f"  {i}/{len(images)} frames", flush=True)
    else:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        for img in images:
            r = model.predict(img, imgsz=a.imgsz, conf=a.conf, classes=[0], device=a.device, verbose=False)[0]
            for box, s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                dets.append((img, s, tuple(box)))
    return dets


def evaluate(weights, name, a):
    gts = load_gt(a.data_dir, a.split)
    dets = predict_all(weights, list(gts), a)

    groups = {"overall": None}
    for img in gts:
        groups.setdefault(condition(img), None)
    result = {"name": name, "mode": a.mode, "weights": str(weights), "frames": len(gts), "settings": {
        "slice": a.slice, "overlap": a.overlap, "merge": a.merge, "full_pass": not a.no_full_pass,
        "conf": a.conf, "iou": a.iou, "imgsz": a.imgsz}, "groups": {}}
    for g in groups:
        keep = {i for i in gts if g == "overall" or condition(i) == g}
        result["groups"][g] = {"frames": len(keep), **metrics(
            [d for d in dets if d[0] in keep], {i: gts[i] for i in keep}, a.iou)}
    return result


def print_table(res):
    print(f"\n{res['name']} [{res['mode']}]  weights: {res['weights']}")
    print(f"{'group':<10}{'frames':>7}{'people':>8}{'AP50':>8}{'P@.25':>8}{'R@.25':>8}{'bestF1':>8}{'@conf':>7}")
    for g, m in res["groups"].items():
        print(f"{g:<10}{m['frames']:>7}{m['n_gt']:>8}{m['AP50']:>8.3f}{m[f'P@{REPORT_CONF}']:>8.3f}"
              f"{m[f'R@{REPORT_CONF}']:>8.3f}{m['best_F1']:>8.3f}{m['best_conf']:>7.2f}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", help="label for the results file (default: weights folder name)")
    ap.add_argument("--mode", choices=["sahi", "plain"], default="sahi")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "uavsar_test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--slice", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--merge", default="GREEDYNMM", choices=["GREEDYNMM", "NMM", "NMS", "LSNMS"])
    ap.add_argument("--no-full-pass", action="store_true", help="skip the extra whole-frame prediction")
    ap.add_argument("--conf", type=float, default=0.05, help="lowest confidence kept; the sweep needs it low")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640, help="plain mode only")
    ap.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "testing" / "results")
    return ap


def main():
    a = build_parser().parse_args()
    weights = Path(a.weights)
    name = a.name or (weights.parent.parent.name if weights.name == "best.pt" else weights.stem)
    res = evaluate(a.weights, name, a)
    print_table(res)
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / f"{name}_{a.mode}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
