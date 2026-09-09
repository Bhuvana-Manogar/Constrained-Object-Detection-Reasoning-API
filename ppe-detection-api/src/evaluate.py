"""
evaluate.py -- Runs evaluation on the held-out test split and dumps
per-class metrics + a confusion matrix image for the memo.

WHY per-class, not just overall mAP (defend this if asked):
An overall mAP@50 of e.g. 0.80 can hide the fact that one class (say
NO-Mask, likely the rarest) is at 0.30 while common classes like Person
and Hardhat carry the average. The brief explicitly wants "confusion
behavior on your classes" -- overall mAP alone doesn't show that, and
self-reported metrics carry less weight than honest analysis of *where*
the model actually fails.
"""
import argparse
from pathlib import Path

from ultralytics import RTDETR


def main(weights: Path, data_yaml: Path, split: str):
    model = RTDETR(str(weights))

    metrics = model.val(data=str(data_yaml), split=split, save_json=True, plots=True)

    print("\n=== OVERALL METRICS ===")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"Precision (mean): {metrics.box.mp:.4f}")
    print(f"Recall (mean):    {metrics.box.mr:.4f}")

    print("\n=== PER-CLASS METRICS (put this table in your memo) ===")
    names = metrics.names
    for i, class_name in names.items():
        try:
            ap50 = metrics.box.ap50[i]
            p = metrics.box.p[i]
            r = metrics.box.r[i]
            print(f"{class_name:20s}  AP50={ap50:.3f}  P={p:.3f}  R={r:.3f}")
        except (IndexError, KeyError):
            print(f"{class_name:20s}  (no instances in this split -- note this in your memo)")

    print("\nConfusion matrix and PR curves saved under runs/detect/val*/ -- "
          "pull confusion_matrix.png directly into your memo, and go through "
          "the highest off-diagonal cells manually to pick your 5 failure cases.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt from training")
    parser.add_argument("--data", type=Path, default=Path("data/data.yaml"))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    main(args.weights, args.data, args.split)
