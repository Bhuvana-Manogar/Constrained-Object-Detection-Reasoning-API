# Written Memo -- Constrained Object Detection & Reasoning API
*(max 2 pages -- trim to fit once filled in)*

**PLACEHOLDERS marked [[FILL IN]] require YOUR actual training run.**
Do not submit this with placeholders still in it, and do not fill them
with invented numbers -- the grading philosophy explicitly rewards honest
lower numbers over polished-looking fake ones.

---

## 1. Domain, dataset, and sourcing

**Domain:** Construction-site PPE (personal protective equipment) compliance
detection.

**Dataset:** Construction Site Safety Image Dataset (Roboflow Universe,
also mirrored on Kaggle by snehilsanyal). ~2,800 images, 10 classes:
Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone,
Safety Vest, machinery, vehicle.

**Why this domain:** at least four of the ten classes (NO-Hardhat, NO-Mask,
NO-Safety Vest, and the presence/absence framing itself) are not standard
COCO categories -- they encode *absence of an object*, which an unmodified
COCO-pretrained model structurally cannot detect. This satisfies the
non-COCO-class requirement without relabeling a COCO class under a new
name. The domain also has an obvious real-world deployment use (automated
safety compliance monitoring) which grounds the "why this domain" question
in something other than convenience.

**Labeling:** dataset was pre-annotated by the original Roboflow
contributors; I did not hand-label from scratch. [[FILL IN: if you added
or re-labeled ANY images yourself, describe exactly what and why. If you
used the dataset as-is, say so plainly -- do not imply hand-labeling you
didn't do.]]

## 2. Train/val/test split strategy

75/15/10, stratified by each image's *rarest* annotated class (see
`src/split_dataset.py`), rather than the dataset's shipped split. Plain
random splitting risks starving rare classes (e.g. NO-Mask) from
validation/test, producing unstable or misleading per-class metrics.
15% (not the more common 10%) was allocated to validation because with
~2,800 images across 10 classes, a smaller val set risks too few instances
of rare classes to get a stable per-class mAP to actually watch during
training.

Seed used: 42 (fixed in both `split_dataset.py` and `train.py` for
reproducibility).

## 3. Evaluation metrics and what they do/don't tell you

[[FILL IN from your actual `evaluate.py` output]]

- Overall mAP50-95: ___
- Overall mAP50: ___
- Per-class AP50/Precision/Recall table: ___ (paste the table `evaluate.py` prints)

**What these metrics tell you:** relative ranking of how well the model
localizes and classifies each PPE category on data drawn from the same
distribution as training.

**What they don't tell you:** (a) performance on the hidden evaluation
set, which may differ in camera angle, lighting, or site type from this
dataset -- self-reported metrics here are not a promise of hidden-set
performance; (b) real-world reliability, since per-person PPE association
isn't done via box overlap in this version (see README limitations) --
mAP measures box/class accuracy, not "did we correctly flag the right
person as non-compliant."

## 4. Five failure cases with root-cause analysis

[[FILL IN -- these must come from YOUR actual test-set predictions.
Go through `runs/ppe_rtdetr/val*/` outputs and the confusion matrix from
`evaluate.py`, pick five real wrong predictions, and for each include:
the image (or a crop), the model's actual output, the ground truth, and
your root-cause hypothesis. Suggested categories to look for, since this
domain naturally produces them:]]

1. **Small/distant object** -- e.g. a hardhat far from camera, below the
   effective receptive field / resolution for reliable detection.
2. **Occlusion** -- workers clustered together, one person's vest
   partially blocked by another's body or equipment.
3. **Class confusion at decision boundary** -- Hardhat vs NO-Hardhat
   predicted with similar confidence when the hat is at an angle that
   obscures whether it's actually a hardhat.
4. **Lighting** -- overexposed/harsh shadow regions common in outdoor
   construction photography degrading feature quality.
5. **Motion blur** -- workers in mid-motion, common in candid site photos.

*(Replace the above with your actual five once you have real predictions
to point to -- generic descriptions without a real example attached will
read as fabricated.)*

## 5. Part B reasoning layer: detector-call decision + insufficient-info example

Implemented as three plain functions (no agentic framework), see
`api/reasoning.py`:

1. `needs_detection(question)` -- keyword/regex classifier deciding whether
   the question is about image content at all (vs. e.g. "what's your name").
2. `reason_over_detections(question, detections)` -- pure Python logic
   handling counting, PPE-violation ("is anyone not wearing X" -- the
   brief's own example question), and most-common-object questions over
   the structured detector output.
3. Confidence guardrail -- if fewer than `MIN_CONFIDENT_DETECTIONS` boxes
   clear `LOW_CONFIDENCE_THRESHOLD` (0.35), the API returns
   `confident: false` and an explicit "not enough evidence" message
   instead of guessing.

**Concrete "insufficient information" example:**
[[FILL IN -- run `/ask` on a genuinely ambiguous/low-quality image from
your test set (blurry, far away, or occluded) and paste the real JSON
response here, e.g.:]]
```
Question: "Is anyone not wearing a safety vest?"
Response: {
  "used_detector": true,
  "answer": "I don't have enough confident detection evidence to answer
             this reliably. (No sufficiently confident detections in this
             image.)",
  "confident": false,
  "raw_detections": [...]
}
```

## 6. Known limitations / what I'd improve with more time

- Person-to-PPE association is at image level, not per-person via box
  overlap/IoU -- a real simplification made under the submission deadline.
- Intent routing is keyword-based, not a learned classifier -- will
  misroute unusual phrasings.
- [[FILL IN anything else you noticed while actually running this]]
