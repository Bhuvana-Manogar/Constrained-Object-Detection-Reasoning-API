"""
verify_cone_stats.py -- computes the REAL, same-split numbers for the
Safety Cone failure case, so the memo doesn't mix validation-set and
test-set statistics (a real error caught in review -- see chat).

Run with:
    python verify_cone_stats.py
"""
from pathlib import Path

def count_instances_by_class(labels_dir: Path, class_id: int) -> tuple[int, int]:
    """Returns (num_images_containing_class, num_instances_of_class)."""
    images_with_class = 0
    total_instances = 0
    for label_file in labels_dir.glob("*.txt"):
        count_in_this_file = 0
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if int(line.split()[0]) == class_id:
                    count_in_this_file += 1
        if count_in_this_file > 0:
            images_with_class += 1
            total_instances += count_in_this_file
    return images_with_class, total_instances


# Safety Cone is class index 6 in data.yaml (0-indexed: Hardhat, Mask,
# NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest,
# machinery, vehicle)
SAFETY_CONE_CLASS_ID = 6

for split_name in ["train", "val", "test"]:
    labels_dir = Path(f"data/split/{split_name}/labels")
    if not labels_dir.exists():
        print(f"{split_name}: folder not found, skipping")
        continue
    images, instances = count_instances_by_class(labels_dir, SAFETY_CONE_CLASS_ID)
    print(f"{split_name}: {images} images contain Safety Cone, {instances} total Safety Cone instances")

print("\n=== USE ONLY THE 'test' ROW ABOVE FOR YOUR FAILURE-CASE ANALYSIS ===")
print("The confusion matrix and evaluate.py output are both from the TEST split.")
print("Do not mix these with the 'val' row numbers (that's what happened before).")
