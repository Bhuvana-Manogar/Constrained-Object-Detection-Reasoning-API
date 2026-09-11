# Constrained PPE Detection & Reasoning API

An object detection and reasoning system for construction-site safety
compliance. Fine-tuned RT-DETR detects personal protective equipment
(PPE) and its absence, exposed through a FastAPI service with a
lightweight natural-language reasoning layer on top.

## Overview

- **Part A** -- an RT-DETR model fine-tuned to detect 10 classes:
  hardhats, masks, safety vests, safety cones, machinery, vehicles, and
  people, including PPE-*absence* classes (NO-Hardhat, NO-Mask,
  NO-Safety Vest).
- **Part B** -- a hand-written reasoning layer that takes a natural-
  language question about an image, decides whether it needs the
  detector, and answers using the detector's structured output --
  including an explicit "not enough evidence" response when detection
  confidence is too low to answer reliably.

## Results

| Metric | Value |
|---|---|
| mAP50 | 0.9048 |
| mAP50-95 | 0.6624 |
| Precision (mean) | 0.9184 |
| Recall (mean) | 0.8652 |

Per-class breakdown, training run details, failure-case analysis, and
model limitations are documented in [`MEMO.md`](MEMO.md).

Full training logs, evaluation curves, and confusion matrix are in
`notebooks/train_on_colab.ipynb` and `results/`.

## Dataset

[Construction Site Safety Image Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)
(Roboflow Universe, mirrored on Kaggle) -- 2,801 images across 10 classes.

Split: 2,101 train / 422 validation / 278 test, using a custom
deterministic strategy (`src/split_dataset.py`) that assigns images by
their rarest present class, with a fixed seed for reproducibility.

## Project structure

```
api/                  FastAPI app, detector wrapper, reasoning layer
src/                  Dataset split, training, and evaluation scripts
data/                 Dataset config (data.yaml) and split data
models/               Trained weights (best.pt)
notebooks/            Colab training notebook with full run history
results/              Confusion matrix and PR/precision/recall/F1 curves
demo.html             Browser-based demo UI for the live API
Dockerfile, docker-compose.yml   Containerized deployment
```

## Setup

```bash
pip install -r requirements.txt
pip install "numpy<2"   # required for ultralytics/opencv compatibility
```

## Training

```bash
python src/split_dataset.py --source data/raw --dest data/split --seed 42
python src/train.py --data data/data.yaml --epochs 100 --imgsz 640 --batch 16 --device 0
python src/evaluate.py --weights models/best.pt --data data/data.yaml --split test
```

No local GPU is required -- `notebooks/train_on_colab.ipynb` runs the
full pipeline on Google Colab's free tier and is already populated with
a complete training run.

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

### `POST /ask`

```bash
curl -X POST -F "image=@path/to/image.jpg" -F "question=Is anyone not wearing a helmet?" http://localhost:8000/ask
```

```json
{
  "used_detector": true,
  "answer": "Yes -- detected violation(s): NO-Safety Vest (1 instance(s)).",
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

1. `needs_detection(question)` -- decides whether a question requires
   image analysis at all.
2. `reason_over_detections(question, detections)` -- handles counting,
   PPE-violation, and most-common-object questions over the detector's
   structured output. PPE-violation classes are treated as image-level
   evidence; the system does not attempt to attribute a violation to a
   specific person (see Limitations).
3. A confidence guardrail that declines to answer when detection
   confidence is insufficient, rather than guessing.

## Limitations

- No person-to-PPE attribution: the system reports that a violation
  class was detected, not which specific person it belongs to.
- Intent routing uses keyword matching, not a learned classifier.
- The confidence threshold (0.35) is a reasonable default, not tuned
  against a precision/recall sweep for this specific task.

Full details and root-cause analysis for model failure cases are in
[`MEMO.md`](MEMO.md).

## License

Built for a technical screening exercise. Dataset used under its
original Roboflow/Kaggle license terms.
