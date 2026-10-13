"""Kaggle version of the full run (model1): baseline score, then train + score each stage.

Edit the CONFIG block below (the dataset names), then run this file in a Kaggle notebook.

Notebook setup:
  1. Settings: Accelerator = GPU (T4 or P100), Internet = On.
  2. Add input: your code dataset (folders training/, testing/, test_runs/, configs/)
     and the data datasets (visdrone_tiled, heridal_val15, sard, uavsar_test).
  3. Cell 1:  !pip install ultralytics sahi
  4. Cell 2:  !python /kaggle/input/<your-code-dataset>/kaggle/run_kaggle.py
     (or paste this file into a cell and run it)
  5. Use "Save & Run All" (commit) so /kaggle/working keeps the outputs.

Everything is written to /kaggle/working/outputs: results.md, results.json and each stage's best.pt.
A session lasts at most about 12 hours, so run long stages one per version (set STAGES) and feed the
earlier weights back in through PREVIOUS_RUNS.
"""
import os
import shutil
import sys
from pathlib import Path

# ============================== CONFIG: change these ==============================
# Folder of the code dataset that contains training/, testing/, test_runs/, configs/.
CODE_INPUT = "/kaggle/input/CHANGE-ME-code"

# Where each data dataset is mounted. The script looks for the folder that holds images/ and labels/,
# up to two levels down, so a zipped-folder nesting level is fine.
DATA_INPUTS = {
    "visdrone_tiled": "/kaggle/input/CHANGE-ME-visdrone-tiled",
    "heridal_val15": "/kaggle/input/CHANGE-ME-heridal-val15",
    "sard": "/kaggle/input/CHANGE-ME-sard",
    "uavsar_test": "/kaggle/input/CHANGE-ME-uavsar-test",
}

STAGES = ["s1", "s2", "s3"]      # s4 needs your own footage and configs/data_own.yaml
EPOCHS = {"s1": 20, "s2": 50, "s3": 50}   # per stage; stage 1 is only pre-training
BATCH = 16                       # T4/P100 have 16 GB; lower it if you see out-of-memory errors
DEVICE = "0"                     # training GPU; scoring uses "cuda:0" below
SCORE_BASELINE = True            # score the COCO yolo11s.pt before training
SCORE_PLAIN = True               # also score without slicing, to see what SAHI adds

# Continuing in a later session: a dataset that holds the earlier runs, laid out as
# <folder>/s1_visdrone/weights/best.pt, <folder>/s2_heridal/weights/best.pt, ...  None = start fresh.
PREVIOUS_RUNS = None             # e.g. "/kaggle/input/CHANGE-ME-previous-runs"
# ==================================================================================

WORK = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/outputs")
CODE_DIRS = ["training", "testing", "test_runs", "configs"]


def find_data_root(path: Path) -> Path:
    """The folder containing images/ and labels/, searched up to two levels down."""
    for depth in (0, 1, 2):
        for p in [path] if depth == 0 else path.glob("/".join(["*"] * depth)):
            if (p / "images").is_dir() and (p / "labels").is_dir():
                return p
    raise FileNotFoundError(f"no images/ + labels/ folder found under {path}")


def setup():
    code = Path(CODE_INPUT)
    if not code.exists():
        raise SystemExit(f"CODE_INPUT not found: {code}. Edit CONFIG at the top of this file.")
    if WORK.exists():
        shutil.rmtree(WORK)
    for d in CODE_DIRS:
        src = code / d if (code / d).exists() else next(code.glob(f"*/{d}"), None)
        if src is None:
            raise SystemExit(f"folder {d}/ not found in {code}")
        shutil.copytree(src, WORK / d)

    (WORK / "datasets").mkdir()
    for name, src in DATA_INPUTS.items():
        # only link what the chosen stages need
        needed = {"visdrone_tiled": "s1", "heridal_val15": "s2", "sard": "s3"}.get(name)
        if needed and needed not in STAGES:
            continue
        root = find_data_root(Path(src))
        (WORK / "datasets" / name).symlink_to(root, target_is_directory=True)
        print(f"data {name} -> {root}")

    if PREVIOUS_RUNS:
        dst = WORK / "runs" / "detect"
        for run in Path(PREVIOUS_RUNS).glob("*/weights/best.pt"):
            target = dst / run.parent.parent.name / "weights"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy(run, target / "best.pt")
            print(f"restored {run.parent.parent.name}")


def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (WORK / "test_runs" / "model1").glob("results.*"):
        shutil.copy(f, OUT / f.name)
    for best in (WORK / "runs" / "detect").glob("*/weights/best.pt"):
        run = best.parent.parent.name
        shutil.copy(best, OUT / f"{run}_best.pt")
    # remove the data links so the notebook output does not drag the datasets along
    shutil.rmtree(WORK / "datasets", ignore_errors=True)
    print(f"\noutputs in {OUT}:")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")


def main():
    setup()
    os.chdir(WORK)
    os.environ["WANDB_MODE"] = "disabled"
    sys.path[:0] = [str(WORK / "training"), str(WORK / "testing"), str(WORK / "test_runs")]

    import train_stages  # same module object that model1 imports

    # stage epochs and batch from CONFIG (STAGES tuple: name, data, epochs, batch, lr0)
    for key, (name, data, _, _, lr0) in list(train_stages.STAGES.items()):
        train_stages.STAGES[key] = (name, data, EPOCHS.get(key, 50), BATCH, lr0)

    import model1

    argv = ["model1.py", "--stages", *STAGES, "--device", DEVICE]
    if not SCORE_BASELINE:
        argv.append("--skip-baseline")
    if not SCORE_PLAIN:
        argv.append("--no-plain")
    sys.argv = argv
    try:
        model1.main()
    finally:
        collect()


if __name__ == "__main__":
    main()
