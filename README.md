# Constrained PPE Detection & Reasoning API

An object detection and reasoning system for construction-site safety
compliance. Fine-tuned RT-DETR detects personal protective equipment
(PPE) and its absence, exposed through a FastAPI service with a
lightweight natural-language reasoning layer on top.

## Overview

- **Part A** -- an RT-DETR model fine-tuned to detect 10 classes,
  including custom PPE-absence classes such as `NO-Hardhat`, `NO-Mask`,
  and `NO-Safety Vest`, which are not standard COCO categories.
- **Part B** -- a hand-written reasoning layer that takes a
  natural-language question about an image, decides whether it needs
  the detector, and answers using the detector's structured output --
  including an explicit "not enough evidence" response when detection
  confidence is too low to answer reliably.

## Classes detected

```
Hardhat
Mask
NO-Hardhat
NO-Mask
NO-Safety Vest
Person
Safety Cone
Safety Vest
machinery
vehicle
```

## Results

| Metric | Value |
|---|---|
| mAP50 | 0.9048 |
| mAP50-95 | 0.6624 |
| Precision (mean) | 0.9184 |
| Recall (mean) | 0.8652 |

Per-class breakdown, training run details, failure-case analysis, and
model limitations are documented in [`MEMO.pdf`](MEMO.pdf)
([source](MEMO.md)).

Full training logs, evaluation curves, and confusion matrix are in
`notebooks/train_on_colab.ipynb` and `results/`.

**Evaluation note:** these metrics are measured on my held-out test set
and are not expected to represent performance on RAP's private hidden
evaluation set. The reported results are provided for reproducibility
and error analysis, not as a claim about hidden-set performance.

## Dataset

[Construction Site Safety Image Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)
(Roboflow Universe, mirrored on Kaggle) -- 2,801 images across the 10
classes above.

Split: 2,101 train / 422 validation / 278 test, using a custom
deterministic strategy (`src/split_dataset.py`) that assigns images by
their rarest present class, with a fixed seed for reproducibility.

## Model weights

The trained RT-DETR-L checkpoint is included in the repository at:

```
models/best.pt
```

The API loads this checkpoint directly when running locally. The
Dockerfile also copies `models/` into the container image during build,
so the containerized version uses the same checkpoint.

## Why RT-DETR-L

RT-DETR-L was selected instead of the larger RT-DETR-X because it
provided a practical accuracy/training-time trade-off on the available
Tesla T4 GPU, while still offering sufficient capacity for the 10-class
detection task.

## Reproducibility

| | |
|---|---|
| Model | RT-DETR-L |
| Framework | Ultralytics |
| Image size | 640 |
| Epochs | 100 |
| Batch size | 16 |
| Device | Tesla T4 |
| Training time | 251.2 minutes |
| Dataset split seed | 42 |

**Install:**
```bash
pip install -r requirements.txt
pip install "numpy<2"   # required for ultralytics/opencv compatibility
```

**Split:**
```bash
python src/split_dataset.py --source data/raw --dest data/split --seed 42
```

**Train:**
```bash
python src/train.py --data data/data.yaml --epochs 100 --imgsz 640 --batch 16 --device 0
```

**Evaluate:**
```bash
python src/evaluate.py --weights models/best.pt --data data/data.yaml --split test
```

A local GPU is not required if the Colab notebook is used. The
notebook runs the training pipeline on a Tesla T4 GPU in Google Colab.
Local inference may also be run on CPU, although performance will be
slower.

## Project structure

```
api/                  FastAPI app, detector wrapper, reasoning layer
src/                  Dataset split, training, and evaluation scripts
data/                 Dataset configuration; split data is generated
                       locally by src/split_dataset.py (not tracked in git)
models/               Trained weights (best.pt)
notebooks/            Colab training notebook with full run history
results/               Confusion matrix and PR/precision/recall/F1 curves
demo.html             Browser-based demo UI for the live API
Dockerfile, docker-compose.yml   Containerized deployment
MEMO.md / MEMO.pdf    Written memo (dataset, evaluation, failure cases)
```

## Running the API

```bash
python -m uvicorn api.main:app --reload
```

Interactive API docs are available at `http://localhost:8000/docs`, or
open `demo.html` directly in a browser for a simple upload-and-ask
interface (CORS is enabled for local use).

### `POST /detect`

```bash
curl -X POST -F "image=@path/to/image.jpg" -F "conf_threshold=0.3" http://localhost:8000/detect
```

```json
{
  "detections": [
    {"class": "Person", "confidence": 0.9957, "box": [427.9, -0.2, 640.0, 409.1]},
    {"class": "NO-Safety Vest", "confidence": 0.9659, "box": [427.7, 186.4, 638.6, 409.1]}
  ]
}
```

Bounding-box coordinates are returned in the detector's image-coordinate
format. Minor values outside the image boundary (as in the example
above) may occur in raw model outputs.

### `POST /ask`

```bash
curl -X POST -F "image=@path/to/image.jpg" -F "question=Is anyone not wearing a helmet?" http://localhost:8000/ask
```

```json
{
  "used_detector": true,
  "answer": "Yes -- detected violation(s): NO-Hardhat (1 instance(s)).",
  "confident": true,
  "raw_detections": [...]
}
```

If detections are too sparse or low-confidence to answer reliably, the
API responds with `"confident": false` and an explanation instead of
guessing.

## Docker

```bash
docker compose up --build
```

## Reasoning layer design

Implemented as three plain Python functions with no agentic framework
(`api/reasoning.py`):

1. `needs_detection(question)` -- decides whether the question requires
   image analysis.
2. `reason_over_detections(question, detections)` -- handles counting,
   PPE-violation, and most-common-object questions over the detector's
   structured output. PPE-violation classes are treated as image-level
   evidence; the system does not attempt to attribute a violation to a
   specific person (see Limitations).
3. A confidence guardrail that declines to answer when no relevant
   detection exceeds the 0.35 confidence threshold, rather than
   guessing.

For people-dependent
questions, the system also requires sufficiently confident Person
evidence.

## Limitations

- No person-to-PPE attribution: the system reports that a violation
  class was detected, not which specific person it belongs to.
- Intent routing uses keyword matching, not a learned classifier.
- The confidence threshold (0.35) is a reasonable default, not tuned
  against a precision/recall sweep for this specific task.
- Safety Cone's dominant weakness is false positives, not misses.

Full details and root-cause analysis for model failure cases are in
[`MEMO.pdf`](MEMO.pdf).

## License

Built for a technical screening exercise. Dataset used under its
original Roboflow/Kaggle license terms.
