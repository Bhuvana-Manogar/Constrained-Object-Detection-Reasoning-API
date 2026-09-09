"""
test_reasoning.py -- Test Part B's decision logic locally, WITHOUT needing
a trained model. This feeds hand-made fake detections into
reason_over_detections() so you can see and understand exactly what the
logic does before you ever touch real model output.

Run with:
    python test_reasoning.py

Read through each test case's OUTPUT and compare it to what you'd expect
-- this is how you build the understanding you'll need to defend this
code verbally, independent of whether training has finished.
"""
import sys
from pathlib import Path

# Allow importing api.reasoning without needing the full package installed
sys.path.insert(0, str(Path(__file__).parent))

from api.reasoning import needs_detection, reason_over_detections


def show(title, result_or_tuple):
    print(f"\n--- {title} ---")
    if isinstance(result_or_tuple, tuple):
        answer, confident = result_or_tuple
        print(f"  answer:    {answer}")
        print(f"  confident: {confident}")
    else:
        print(f"  {result_or_tuple}")


# ============================================================
# TEST 1: Intent routing -- questions that should NOT trigger detection
# ============================================================
print("=" * 60)
print("TEST 1: Intent routing -- non-image questions")
print("=" * 60)
for q in ["What's your name?", "Hello!", "Who built you?", "What can you do?"]:
    show(q, needs_detection(q))  # expect False for all of these

# ============================================================
# TEST 2: Intent routing -- questions that SHOULD trigger detection
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Intent routing -- image-content questions")
print("=" * 60)
for q in [
    "How many people are in this image?",
    "Is anyone not wearing a helmet?",
    "What's the most common object here?",
]:
    show(q, needs_detection(q))  # expect True for all of these

# ============================================================
# TEST 3: Reasoning over FAKE detections -- a clean, confident case
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Clear violation -- one person, no hardhat")
print("=" * 60)
fake_detections_violation = [
    {"class": "Person", "confidence": 0.91, "box": [10, 10, 100, 200]},
    {"class": "NO-Hardhat", "confidence": 0.87, "box": [20, 10, 60, 40]},
    {"class": "Safety Vest", "confidence": 0.76, "box": [15, 80, 90, 150]},
]
show(
    "Is anyone not wearing a helmet?",
    reason_over_detections("Is anyone not wearing a helmet?", fake_detections_violation),
)
# Expected: answer mentions NO-Hardhat found, confident=True

# ============================================================
# TEST 4: Reasoning over FAKE detections -- everyone compliant
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: Fully compliant -- hardhat + vest, no violations")
print("=" * 60)
fake_detections_compliant = [
    {"class": "Person", "confidence": 0.90, "box": [10, 10, 100, 200]},
    {"class": "Hardhat", "confidence": 0.88, "box": [20, 10, 60, 40]},
    {"class": "Safety Vest", "confidence": 0.81, "box": [15, 80, 90, 150]},
]
show(
    "Is anyone not wearing a helmet?",
    reason_over_detections("Is anyone not wearing a helmet?", fake_detections_compliant),
)
# Expected: "No PPE violations detected...", confident=True

# ============================================================
# TEST 5: Counting question
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: Counting -- 'how many people'")
print("=" * 60)
fake_detections_crowd = [
    {"class": "Person", "confidence": 0.9, "box": [0, 0, 10, 10]},
    {"class": "Person", "confidence": 0.85, "box": [20, 0, 30, 10]},
    {"class": "Person", "confidence": 0.6, "box": [40, 0, 50, 10]},
    {"class": "Hardhat", "confidence": 0.7, "box": [0, 0, 5, 5]},
]
show(
    "How many people are in this image?",
    reason_over_detections("How many people are in this image?", fake_detections_crowd),
)
# Expected: count of 3 (Person instances), confident=True

# ============================================================
# TEST 6: THE CONFIDENCE GUARDRAIL -- this is your memo's key example
# ============================================================
print("\n" + "=" * 60)
print("TEST 6: Guardrail -- low-confidence / sparse detections")
print("=" * 60)
fake_detections_uncertain = [
    {"class": "Person", "confidence": 0.20, "box": [0, 0, 5, 5]},  # below LOW_CONFIDENCE_THRESHOLD
]
show(
    "Is anyone not wearing a safety vest?",
    reason_over_detections("Is anyone not wearing a safety vest?", fake_detections_uncertain),
)
# Expected: confident=False, "not enough confident detection evidence" message
# THIS is the kind of output that goes in your memo as the required
# "insufficient information" example -- except with a REAL image once
# your model is trained, not this fake one.

# ============================================================
# TEST 7: Guardrail -- completely empty detections
# ============================================================
print("\n" + "=" * 60)
print("TEST 7: Guardrail -- nothing detected at all")
print("=" * 60)
show(
    "What's the most common object here?",
    reason_over_detections("What's the most common object here?", []),
)
# Expected: confident=False

print("\n" + "=" * 60)
print("Done. Read through each 'answer'/'confident' pair above and make")
print("sure you can explain WHY the logic produced that result -- that's")
print("exactly what the verbal defense round will probe.")
print("=" * 60)
