"""model1: the full run. Score the COCO baseline, then for each stage train it and score it.

    baseline (yolo11s.pt) -> s1 VisDrone -> s2 HERIDAL   (s3 SARD and s4 own footage are opt-in)

Every score is on the fixed test set (datasets/uavsar_test), with SAHI sliced inference
and, as a baseline, plain whole-frame inference. Results go to test_runs/model1/.

Setup (run once, in your environment):
    pip install ultralytics sahi

Usage (from the repo root):
    python test_runs/model1.py                          # baseline, then the default stages (s1, s2)
    python test_runs/model1.py --stages s1 s2           # only some stages
    python test_runs/model1.py --skip-train             # only score weights that already exist
    python test_runs/model1.py --skip-baseline --no-plain
    python test_runs/model1.py --epochs 3 --stages s1   # quick trial

Outputs: test_runs/model1/results.json (raw numbers) and results.md (table, paste into docs/results.md).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "model1"
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "testing")]

import sahi_eval  # noqa: E402
import train_stages  # noqa: E402


def score(weights, name, mode, device):
    args = ["--weights", str(weights), "--mode", mode]
    if device:  # training takes "0", SAHI wants "cuda:0"
        args += ["--device", f"cuda:{device}" if device.isdigit() else device]
    a = sahi_eval.build_parser().parse_args(args)
    res = sahi_eval.evaluate(str(weights), name, a)
    sahi_eval.print_table(res)
    return res


def write_report(results):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    conds = [g for g in results[0]["groups"] if g != "overall"] if results else []
    head = "| model | mode | AP50 | P@.25 | R@.25 | best F1 | " + " | ".join(f"AP50 {c}" for c in conds) + " |"
    rows = [head, "|" + "---|" * (6 + len(conds))]
    for r in results:
        o = r["groups"]["overall"]
        rows.append(f"| {r['name']} | {r['mode']} | {o['AP50']:.3f} | {o['P@0.25']:.3f} | {o['R@0.25']:.3f} | "
                    f"{o['best_F1']:.3f} | " + " | ".join(f"{r['groups'][c]['AP50']:.3f}" for c in conds) + " |")
    (OUT / "results.md").write_text("\n".join(rows) + "\n")
    print(f"\nsaved {OUT / 'results.md'} and results.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=list(train_stages.STAGES),
                    help=f"default: {' '.join(train_stages.DEFAULT_STAGES)}")
    ap.add_argument("--epochs", type=int, help="override epochs for every stage")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device", help="e.g. 0 or cpu for training, cuda:0 or cpu for scoring")
    ap.add_argument("--skip-baseline", action="store_true", help="do not score the COCO weights")
    ap.add_argument("--skip-train", action="store_true", help="do not train, score the newest existing best.pt")
    ap.add_argument("--no-plain", action="store_true", help="skip the plain whole-frame score")
    a = ap.parse_args()

    stages = a.stages or train_stages.DEFAULT_STAGES
    modes = ["sahi"] if a.no_plain else ["sahi", "plain"]
    results = []

    if not a.skip_baseline:
        for mode in modes:
            results.append(score(train_stages.BASE_WEIGHTS, "coco_baseline", mode, a.device))
        write_report(results)

    for i, key in enumerate(stages):
        name = train_stages.STAGES[key][0]
        if a.skip_train:
            best = train_stages.find_best(name)
            if best is None:
                print(f"no weights for {key}, skipping")
                continue
        else:
            best = train_stages.train_stage(key, epochs=a.epochs, batch=a.batch, device=a.device)
        for mode in modes:
            results.append(score(best, name, mode, a.device))
        write_report(results)  # saved after every stage, so a crash keeps earlier results


if __name__ == "__main__":
    main()
