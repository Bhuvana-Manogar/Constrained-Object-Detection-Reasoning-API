# Constrained PPE Detection & Reasoning API

RT-DETR fine-tuned on construction-site PPE compliance, exposed via FastAPI,
with a hand-written natural-language reasoning layer on top.

## 1. Dataset

**Construction Site Safety Image Dataset** (Roboflow Universe, mirrored on
Kaggle by snehilsanyal). ~2,800 images, 10 classes:
`Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone,
Safety Vest, machinery, vehicle`.

Download it, export in YOLOv8/YOLO-txt format, and place under:
```
data/raw/images/*.jpg
data/raw/labels/*.txt
```

## 2. Setup

```bash
pip install -r requirements.txt
```

Record your actual environment for reproducibility:
```bash
pip freeze > environment_lock.txt
nvidia-smi > gpu_info.txt   # or note "CPU only" / Colab GPU tier
```

## 3. Split, train, evaluate

```bash
python src/split_dataset.py --source data/raw --dest data/split --seed 42
python src/train.py --data data/data.yaml --epochs 100 --imgsz 640 --batch 16 --device 0
python src/evaluate.py --weights runs/ppe_rtdetr/weights/best.pt --data data/data.yaml --split test
```

No local GPU? Use `notebooks/train_on_colab.ipynb` -- push this repo to
GitHub first, then clone it inside Colab.

After training, copy the weights so the API can find them:
```bash
cp runs/ppe_rtdetr/weights/best.pt models/best.pt
```

## 4. Run the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### POST /detect
```bash
curl -X POST -F "image=@sample.jpg" -F "conf_threshold=0.3" http://localhost:8000/detect
```
```json
{
  "detections": [
    {"class": "Hardhat", "confidence": 0.91, "box": [120.0, 45.0, 210.0, 160.0]},
    {"class": "NO-Safety Vest", "confidence": 0.67, "box": [300.0, 80.0, 400.0, 260.0]}
  ]
}
```

### POST /ask
```bash
curl -X POST -F "image=@sample.jpg" -F "question=Is anyone not wearing a helmet?" \
     http://localhost:8000/ask
```
```json
{
  "used_detector": true,
  "answer": "Yes -- detected violation(s): NO-Hardhat (1 instance(s)).",
  "confident": true,
  "raw_detections": [...]
}
```

Example of the confidence guardrail firing (low-confidence/empty detections):
```json
{
  "used_detector": true,
  "answer": "I don't have enough confident detection evidence to answer this reliably. (No sufficiently confident detections in this image.)",
  "confident": false,
  "raw_detections": []
}
```

## 5. Things you must be able to defend verbally

These are the exact decisions baked into this repo -- know the *why*, not
just the *what*, for each:

1. **Why this dataset/domain** -- non-COCO classes are structural
   (presence/absence of PPE), not just relabeled COCO categories; real-world
   deployment story (compliance monitoring).
2. **Why 75/15/10, stratified by rarest class per image** (`split_dataset.py`
   docstring) -- rather than the dataset's shipped split.
3. **Why fine-tune from COCO-pretrained RT-DETR rather than train from
   scratch** (`train.py` docstring) -- backbone transfer, dataset size.
4. **Why per-class metrics, not just overall mAP** (`evaluate.py` docstring).
5. **Why no agentic framework in Part B, and what the three-function
   decision layer actually does** (`reasoning.py` docstring) --
   `needs_detection()` -> `detect()` -> `reason_over_detections()`.
6. **The specific "insufficient information" example** -- point to the
   guardrail JSON example above and be ready to explain *why* it triggered
   (confidence below `LOW_CONFIDENCE_THRESHOLD`, or zero trustworthy boxes).
7. **Five failure cases** -- these come from YOUR actual evaluation run, not
   from this repo. You must generate these yourself by looking at real
   predictions on your test set (see `evaluate.py` output).

## 6. Known limitations (say these out loud, don't hide them)

- `needs_detection()` is keyword/regex-based -- it will misroute unusual
  phrasings (sarcasm, indirect references, multi-part questions).
- `reason_over_detections()` has explicit rules for count/violation/most-common
  questions; anything else falls through to a raw summary flagged as
  low-confidence, not a targeted answer.
- Person-to-PPE association is NOT done by box overlap/IoU in this version --
  violations are reported at the image level ("a NO-Hardhat box exists
  somewhere"), not per-identified-person. This is a real simplification --
  acknowledge it as a design trade-off (time-boxed submission) rather than a
  hidden gap, and it's a good thing to mention as a "if I had more time"
  improvement in your memo.
