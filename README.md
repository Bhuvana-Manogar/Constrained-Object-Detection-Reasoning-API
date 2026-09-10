# Constrained PPE Detection & Reasoning API

RT-DETR fine-tuned on construction-site PPE compliance, exposed via FastAPI,
with a hand-written natural-language reasoning layer on top.

**Status: trained, evaluated, and tested live end-to-end.** See
`MEMO_HUMANIZED.md` for the full writeup with real results.

## 1. Real results (from actual training + evaluation)

- **mAP50: 0.9048, mAP50-95: 0.6624** on a held-out test split (278 images)
- Trained: RT-DETR-l, 100 epochs, 640px, batch 16, Tesla T4, 251.2 minutes
- Weakest class: Safety Cone (0.764 AP50, 0.637 recall) -- see memo Section 4
  for root-cause analysis
- Full training log with all epoch output: `notebooks/train_on_colab.ipynb`
  (rendered directly on GitHub)
- Confusion matrix + PR/P/R/F1 curves: `results/`
- Trained weights: `models/best.pt` (66.3MB, tracked in git)

## 2. Dataset

**Construction Site Safety Image Dataset** (Roboflow Universe, mirrored on
Kaggle by snehilsanyal). 2,801 images, 10 classes:
`Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone,
Safety Vest, machinery, vehicle`.

Re-split 75/15/10 (own stratified split, not the dataset's shipped one) --
see `src/split_dataset.py` and memo Section 2 for reasoning.

## 3. Setup

```bash
pip install -r requirements.txt
pip install "numpy<2"   # required: ultralytics/opencv needs NumPy 1.x
```

## 4. Reproducing training

```bash
python src/split_dataset.py --source data/raw --dest data/split --seed 42
python src/train.py --data data/data.yaml --epochs 100 --imgsz 640 --batch 16 --device 0
python src/evaluate.py --weights models/best.pt --data data/data.yaml --split test
```

No local GPU? Use `notebooks/train_on_colab.ipynb` in Google Colab --
already contains the full run with real output, viewable directly on GitHub.

## 4. Run the API

```bash
python -m uvicorn api.main:app --reload
```

Then either use curl, or open the interactive test UI in your browser at:
```
http://localhost:8000/docs
```
(Opening `http://localhost:8000/detect` or `/ask` directly in a browser
will show "Not Found" / "Method Not Allowed" -- that's expected, since
those are POST-only endpoints. Use `/docs` or curl instead.)

### POST /detect -- real example, from the live API
```bash
curl.exe -X POST -F "image=@data/split/test/images/youtube-824_jpg.rf.66d684a9888bb29ffe83793dd2ed3528.jpg" -F "conf_threshold=0.3" http://localhost:8000/detect
```
```json
{"detections":[{"class":"Person","confidence":0.9957,"box":[427.9,-0.2,640.0,409.1]},{"class":"NO-Safety Vest","confidence":0.9659,"box":[427.7,186.4,638.6,409.1]}, ...]}
```

### POST /ask -- real example, confident answer
```bash
curl.exe -X POST -F "image=@data/split/test/images/youtube-824_jpg.rf.66d684a9888bb29ffe83793dd2ed3528.jpg" -F "question=Is anyone not wearing a helmet?" http://localhost:8000/ask
```
```json
{"used_detector":true,"answer":"Yes -- detected violation(s): NO-Safety Vest (1 instance(s)).","confident":true, ...}
```

### POST /ask -- real example, guardrail firing (insufficient information)
```bash
curl.exe -X POST -F "image=@data/split/test/images/airport_inside_0073_jpg.rf.405ba1048616d99ee5168b6affb4938e.jpg" -F "question=Is anyone not wearing a safety vest?" http://localhost:8000/ask
```
```json
{"used_detector":true,"answer":"I don't have enough confident detection evidence to answer this reliably. (No people detected in the image with sufficient confidence to answer.)","confident":false, ...}
```

## 5. Docker (bonus)

```bash
docker compose up --build
```

## 6. Things to be able to defend verbally

1. **Why this dataset/domain** -- non-COCO classes are structural
   (presence/absence of PPE), not just relabeled COCO categories.
2. **Why 75/15/10, stratified by rarest class per image** -- rather than
   the dataset's shipped split.
3. **Why fine-tune from COCO-pretrained RT-DETR rather than train from
   scratch** -- backbone transfer, dataset size (2,801 images).
4. **Why per-class metrics matter** -- Safety Cone (0.764 AP50) is hidden
   by the strong overall mAP50 (0.9048) unless you look class-by-class.
5. **Why no agentic framework in Part B** -- three plain functions:
   `needs_detection()` -> `detect()` -> `reason_over_detections()`.
6. **The real "insufficient information" example** -- see Section 4 above,
   and be ready to explain why it triggered (no Person detections found).
7. **A real limitation found through testing, not guessed** -- asking
   about "helmet" instead of "hardhat" doesn't route to the specific
   Hardhat-violation check, since "helmet" isn't in the synonym list.
   See memo Section 6.

## 7. Known limitations (stated openly, not hidden)

- Person-to-PPE association is at image level, not per-person via box
  overlap -- a real simplification made under the deadline.
- Intent routing is keyword-based -- will misroute unusual phrasings.
- Confidence threshold (0.35) is a reasonable default, not swept/tuned
  against a precision-recall trade-off for this specific task.
- Safety Cone detection is the model's clearest weakness (see memo).
