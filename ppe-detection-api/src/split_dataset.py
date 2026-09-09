"""
split_dataset.py

WHY THIS FILE EXISTS (defend this if asked):
The Roboflow export ships with its own train/val/test folders, but those were
split by Roboflow's own logic, not something you chose and can justify. The
brief explicitly grades "train/val/test split strategy and technical
justification" -- so we re-split ourselves and document exactly how.

STRATEGY CHOSEN: stratified split by *image-level dominant class*, 75/15/10.
- Plain random splitting on object-detection data can silently starve rare
  classes (e.g. NO-Mask) from val/test, making your val mAP look better or
  worse than it really is purely by luck of the draw.
- We approximate stratification at the image level: each image is bucketed
  by whichever of its annotated classes is rarest (since that's the class
  most at risk of being under-represented), then we split within each bucket.
- 75/15/10 rather than 80/10/10: with ~2,800 images and 10 classes, a 10%
  val split risks too few instances of rare classes to get a stable mAP
  during training-time validation. 15% gives more stable per-class metrics
  to actually watch while training; test stays at 10% held fully out.

Run:
    python src/split_dataset.py --source data/raw --dest data/split --seed 42
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def read_yolo_classes(label_path: Path):
    """Return the set of class ids present in one YOLO .txt label file."""
    classes = set()
    if not label_path.exists():
        return classes
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            classes.add(int(line.split()[0]))
    return classes


def main(source: Path, dest: Path, seed: int, train_frac: float, val_frac: float):
    random.seed(seed)

    images_dir = source / "images"
    labels_dir = source / "labels"
    image_paths = sorted(
        [p for p in images_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}. Check --source path.")

    # Bucket each image by its rarest class (approximate stratification).
    class_counts = defaultdict(int)
    image_to_classes = {}
    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        classes = read_yolo_classes(label_path)
        image_to_classes[img_path] = classes
        for c in classes:
            class_counts[c] += 1

    def rarest_class(classes):
        if not classes:
            return -1  # bucket for background/unlabeled images
        return min(classes, key=lambda c: class_counts[c])

    buckets = defaultdict(list)
    for img_path, classes in image_to_classes.items():
        buckets[rarest_class(classes)].append(img_path)

    splits = {"train": [], "val": [], "test": []}
    for bucket_key, imgs in buckets.items():
        imgs = imgs[:]
        random.shuffle(imgs)
        n = len(imgs)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        splits["train"].extend(imgs[:n_train])
        splits["val"].extend(imgs[n_train:n_train + n_val])
        splits["test"].extend(imgs[n_train + n_val:])

    for split_name, imgs in splits.items():
        (dest / split_name / "images").mkdir(parents=True, exist_ok=True)
        (dest / split_name / "labels").mkdir(parents=True, exist_ok=True)
        for img_path in imgs:
            label_path = labels_dir / (img_path.stem + ".txt")
            shutil.copy(img_path, dest / split_name / "images" / img_path.name)
            if label_path.exists():
                shutil.copy(label_path, dest / split_name / "labels" / label_path.name)

    print("Split complete.")
    for split_name, imgs in splits.items():
        print(f"  {split_name}: {len(imgs)} images")
    print(f"Seed used: {seed} (record this in your memo for reproducibility)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Folder with images/ and labels/ subfolders")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()
    main(args.source, args.dest, args.seed, args.train_frac, args.val_frac)
