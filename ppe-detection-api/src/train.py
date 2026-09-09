"""
train.py -- Fine-tunes RT-DETR on the PPE dataset.

WHY Ultralytics' RT-DETR implementation (defend this if asked):
The brief allows "any faithful implementation/library" of RT-DETR. Ultralytics'
port is used here because (a) it gives a training/val/predict API consistent
with what most reviewers will expect to read, (b) it handles COCO-format
mAP computation out of the box so Part A evaluation is standardized rather
than hand-rolled and error-prone, and (c) it lets us start from RT-DETR
COCO-pretrained weights and fine-tune, rather than training from scratch --
which matters a lot given a 2-3k image dataset and limited compute/time.

WHY fine-tune from COCO-pretrained rather than random init (defend this):
RT-DETR's backbone (ResNet/HGNetv2) already knows general visual features
(edges, textures, object-ness) from COCO. Random init on ~2-3k images would
almost certainly underfit or take far more epochs to converge. Fine-tuning
only needs to adapt the final classification/regression heads (and lightly
adjust the backbone) to the new 10-class PPE schema.

Record every hyperparameter printed at the end of a run -- reproducibility
is capped at 50% if a reviewer can't reproduce your training from your memo.
"""
import argparse
import time
from pathlib import Path

from ultralytics import RTDETR


def main(data_yaml: Path, epochs: int, imgsz: int, batch: int, model_size: str, device: str):
    # rtdetr-l.pt = "large" backbone, COCO-pretrained. rtdetr-x.pt is the bigger
    # variant if you have the GPU memory and want a small accuracy bump.
    weights = f"rtdetr-{model_size}.pt"
    model = RTDETR(weights)

    start = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs",
        name="ppe_rtdetr",
        seed=42,          # match the split seed for full reproducibility
        patience=15,       # early-stop if val mAP plateaus, avoid overfitting on ~2-3k images
        val=True,
    )
    elapsed_min = (time.time() - start) / 60

    print("\n=== RECORD THIS IN YOUR MEMO ===")
    print(f"Base weights: {weights}")
    print(f"Epochs requested: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print(f"Wall-clock training time: {elapsed_min:.1f} minutes")
    print("Also record your actual GPU model (e.g. `nvidia-smi` output) and")
    print("library versions (`pip freeze > requirements.txt`) for the reproducibility section.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/data.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model-size", choices=["l", "x"], default="l",
                         help="l = rtdetr-l (faster, less VRAM), x = rtdetr-x (more accurate, needs more VRAM)")
    parser.add_argument("--device", default="0", help="'0' for first GPU, 'cpu' if no GPU available")
    args = parser.parse_args()
    main(args.data, args.epochs, args.imgsz, args.batch, args.model_size, args.device)
