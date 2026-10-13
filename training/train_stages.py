"""Staged fine-tuning of YOLO11s: VisDrone -> HERIDAL -> SARD -> own footage.

Each stage starts from the previous stage's best.pt. Keep imgsz the same in every stage.

Setup (run once, in your environment):
    pip install ultralytics sahi

Usage (from the repo root):
    python training/train_stages.py                      # every stage whose config exists
    python training/train_stages.py --stages s1 s2       # only some stages
    python training/train_stages.py --stages s2 --init runs/detect/s1_visdrone/weights/best.pt
    python training/train_stages.py --smoke              # 3 epochs on stage 1, to check paths and labels

Stage names: s1 (visdrone), s2 (heridal), s3 (sard), s4 (own footage, needs configs/data_own.yaml).
Weights land in runs/detect/<stage name>/weights/best.pt.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "runs" / "detect"

# key -> (run name, data config, epochs, batch, lr0). lr0 None = Ultralytics default.
# Each stage after s1 starts from the stage listed in PREVIOUS.
STAGES = {
    "s1": ("s1_visdrone", "configs/data_visdrone_tiled.yaml", 50, 16, None),
    "s2": ("s2_heridal", "configs/data_heridal_val15.yaml", 50, 16, 0.002),
    "s3": ("s3_sard", "configs/data_sard.yaml", 50, 16, 0.002),
    "s4": ("s4_own", "configs/data_own.yaml", 50, 16, 0.001),
}
PREVIOUS = {"s2": "s1", "s3": "s2", "s4": "s3"}
BASE_WEIGHTS = "yolo11s.pt"  # COCO-pretrained, downloaded by Ultralytics on first use


def find_best(run_name: str) -> Path | None:
    """Newest runs/detect/<run_name>*/weights/best.pt (Ultralytics adds a number if the name exists)."""
    found = sorted(PROJECT.glob(f"{run_name}*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def absolute_data_yaml(data: Path, run_name: str) -> Path:
    """Ultralytics may resolve a relative 'path:' against its own datasets folder, not the working
    directory. Write a copy of the config with an absolute path so it always finds the data."""
    import yaml

    cfg = yaml.safe_load(data.read_text())
    if not Path(cfg["path"]).is_absolute():
        cfg["path"] = str((ROOT / cfg["path"]).resolve())
    out = ROOT / "runs" / "data_yamls" / f"{run_name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg))
    return out


def train_stage(key: str, imgsz: int = 640, epochs: int | None = None, batch: int | None = None,
                init: str | Path | None = None, device: str | None = None) -> Path:
    """Train one stage and return the path of its best.pt."""
    from ultralytics import YOLO

    name, data, def_epochs, def_batch, lr0 = STAGES[key]
    if not (ROOT / data).exists():
        raise FileNotFoundError(f"{data} not found, cannot train {key}")

    if init is None:
        if key == "s1":
            init = BASE_WEIGHTS
        else:
            prev = find_best(STAGES[PREVIOUS[key]][0])
            if prev is None:
                raise FileNotFoundError(f"no best.pt for {PREVIOUS[key]} yet, train it first or pass --init")
            init = prev

    args = dict(data=str(absolute_data_yaml(ROOT / data, name)), imgsz=imgsz, epochs=epochs or def_epochs, batch=batch or def_batch,
                project=str(PROJECT), name=name)
    if lr0 is not None:
        # with the default optimizer="auto", Ultralytics ignores lr0 and picks its own rate
        args["lr0"], args["optimizer"] = lr0, "SGD"
    if device:
        args["device"] = device

    print(f"\n=== {key}: {name} | start from {init} | data {data} | {args['epochs']} epochs ===", flush=True)
    model = YOLO(str(init))
    model.train(**args)
    best = Path(model.trainer.best)
    print(f"{key} done, best weights: {best}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=list(STAGES), help="default: all stages with an existing config")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, help="override epochs for every chosen stage")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--init", help="start weights for the FIRST chosen stage (default: previous stage's best.pt)")
    ap.add_argument("--device", help="e.g. 0 or cpu")
    ap.add_argument("--smoke", action="store_true", help="3 epochs, batch 8, stage 1 only")
    a = ap.parse_args()

    if a.smoke:
        a.stages, a.epochs, a.batch = ["s1"], 3, 8
    stages = a.stages or [k for k, v in STAGES.items() if (ROOT / v[1]).exists()]
    for i, key in enumerate(stages):
        train_stage(key, a.imgsz, a.epochs, a.batch, a.init if i == 0 else None, a.device)


if __name__ == "__main__":
    main()
