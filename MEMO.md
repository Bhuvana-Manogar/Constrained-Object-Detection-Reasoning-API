# Written Memo -- Constrained PPE Detection & Reasoning API

## 1. Why this domain and dataset

I chose construction-site PPE (personal protective equipment) compliance
detection. My dataset is the Construction Site Safety Image Dataset from
Roboflow Universe (mirrored on Kaggle by snehilsanyal) -- 2,801 images
across 10 classes: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest,
Person, Safety Cone, Safety Vest, machinery, and vehicle.

I picked this domain for two reasons. First, at least three of the ten
classes -- NO-Hardhat, NO-Mask, NO-Safety Vest -- aren't standard COCO
categories. They encode the *absence* of PPE, which a pretrained COCO
model has no way to detect, so I couldn't just run an off-the-shelf model
and call it done. Second, it has an obvious real-world use: flagging
missing safety gear on a job site in near real time is something that
actually matters.

I didn't hand-label anything new -- the dataset came pre-annotated from
Roboflow. Since this is a fairly well-known public dataset, I wanted my
submission to reflect my own work rather than just downloading and
training on it as-is, so I redid the train/val/test split myself instead
of using the one that shipped with it (details below).

## 2. My split strategy

I went with 75/15/10, stratified by each image's *rarest* class (see
`src/split_dataset.py`), rather than reusing the dataset's original split.
My reasoning: a plain random split can easily starve a rare class like
Safety Cone -- which only has 98 images despite 607 total instances --
out of the validation or test set, which would make its metrics
unreliable. I picked 15% for validation instead of the more typical 10%
because with only ~2,800 images spread across 10 classes, a smaller
validation set risks having too few examples of the rarer classes to get
a stable per-class mAP while training.

The actual split came out to 2,101 train / 422 validation / 278 test
images, with seed 42 fixed in both `split_dataset.py` and `train.py` so
the whole thing is reproducible.

## 3. What my evaluation actually showed

Here's what I got on the held-out test set (278 images the model never
saw during training or validation):

- Overall mAP50-95: **0.6624**
- Overall mAP50: **0.9048**
- Precision (mean): 0.9184
- Recall (mean): 0.8652

Per-class breakdown:

| Class | AP50 | Precision | Recall |
|---|---|---|---|
| Hardhat | 0.860 | 0.906 | 0.803 |
| Mask | 0.939 | 0.932 | 0.919 |
| NO-Hardhat | 0.905 | 0.913 | 0.897 |
| NO-Mask | 0.893 | 0.910 | 0.874 |
| NO-Safety Vest | 0.939 | 0.940 | 0.903 |
| Person | 0.956 | 0.952 | 0.912 |
| Safety Cone | **0.764** | 0.878 | **0.637** |
| Safety Vest | 0.937 | 0.937 | 0.906 |
| machinery | 0.973 | 0.950 | 0.946 |
| vehicle | 0.882 | 0.868 | 0.855 |

I fine-tuned RT-DETR-l from COCO-pretrained weights for 100 epochs at
640px, batch size 16, on a single Tesla T4 (Google Colab). The whole run
took 251.2 minutes.

What jumps out immediately is that Safety Cone is clearly my weakest
class -- 0.764 AP50 and only 0.637 recall, well behind everything else.
That tells me the model is missing more than a third of the real cones
in my test set.

What these numbers *don't* tell me: how the model will actually perform
on RAP's hidden evaluation set, since that data could look different in
lighting, angle, or site type than what I trained on. They also don't
tell me whether the system correctly attributes a specific violation to
a specific worker -- my current design doesn't do that (more on this in
Section 6).

## 4. Five things my model gets wrong, and why

I pulled these from the confusion matrix on my test set
(`results/confusion_matrix.png`):

**1. Safety Cone gets missed a lot.** 348 cones were correctly caught,
but 196 real cones (36%) were missed entirely and read as background
instead. My guess: cones are small and visually simple, and this class
has the fewest training images (98) of any class despite having 607
total instances -- so the model just hasn't seen enough variety of cone
angles and distances to generalize well.

**2. Person detection struggles in crowded scenes.** 910 people were
caught correctly, but 201 were missed. Since Person is my best-represented
class by far (1,443 instances), I don't think this is a data problem --
it's more likely occlusion, where workers get partially hidden behind
machinery, vehicles, or each other in busy site photos.

**3. The model sometimes "sees" cones that aren't there.** Background was
misread as Safety Cone 69 times -- the highest false-positive count of
any class. My best guess is visual similarity to other small,
brightly-colored objects on a site -- warning signage, stacked materials,
that kind of thing.

**4. Same pattern with Person, less often.** Background was misread as
Person 49 times. Could be partial human shapes -- mannequins, reflections,
heavily occluded limbs -- that look "person-like" enough to fool it.

**5. Cone recall stays weak even when I loosen the confidence bar.**
Looking at the Recall-Confidence curve (`results/BoxR_curve.png`), Safety
Cone's line sits noticeably below every other class across the whole
range, capping out around 0.65-0.7 recall even at low thresholds, where
other classes reach 0.85+ easily. That tells me this isn't just a
threshold-tuning issue -- the model genuinely struggles to detect cones
in the first place.

## 5. How my reasoning layer decides when to call the detector

I built this as three plain Python functions -- no framework, per the
brief's rules (see `api/reasoning.py`):

1. `needs_detection(question)` -- a keyword/regex check that decides if a
   question is even about the image at all.
2. `reason_over_detections(question, detections)` -- plain Python logic
   handling counting, PPE-violation questions (including the exact example
   from the brief, "is anyone not wearing a helmet?"), and most-common-
   object questions.
3. A confidence guardrail -- if fewer than one detection clears a 0.35
   confidence threshold, or if a question needs people to be present and
   none are found confidently, I return `confident: false` with an honest
   explanation instead of guessing.

Here's a real example I captured directly from my running API, using my
actual trained weights, not a made-up one:

```
Question: "Is anyone not wearing a safety vest?"
Image: airport_inside_0073_jpg.rf.405ba1048616d99ee5168b6affb4938e.jpg
       (a non-construction image included in the dataset as a negative
       example)
Response: {
  "used_detector": true,
  "raw_detections": [5 "machinery" detections, 0 "Person" detections],
  "answer": "I don't have enough confident detection evidence to answer
             this reliably. (No people detected in the image with
             sufficient confidence to answer.)",
  "confident": false
}
```

The detector found some machinery-like objects but no people at all.
Since the question is asking about people's PPE compliance, my system
correctly recognized it had nothing to go on and said so, instead of
guessing based on detections that had nothing to do with the question.

## 6. What I know is still weak, and what I'd fix with more time

- **I'm not linking specific PPE violations to specific people.** Right
  now my system can say "there's a NO-Safety-Vest box somewhere in this
  image," but not "worker #2 specifically is missing their vest." With
  more time I'd match each PPE box to its nearest Person box using IoU so
  the answer can point to an actual person.
- **My intent routing is just keyword matching, not a learned model.** I
  found this out directly while testing: when I asked "is anyone not
  wearing a *helmet*?" my system answered about a NO-Safety-Vest
  violation instead, because "helmet" isn't mapped as a synonym for
  "Hardhat" in my keyword list. The answer wasn't wrong -- a violation
  really was found -- but it wasn't checking specifically for the thing I
  asked about. That's a real gap I'd fix by expanding the synonym list.
- **Safety Cone is my clearest weak spot** (0.764 AP50, 0.637 recall).
  Given more time, I'd look at adding more cone-specific training images
  or trying a different input resolution for that class.
- **My 0.35 confidence threshold is a reasonable starting point, not
  something I actually tuned.** I didn't sweep it against the
  precision/recall trade-off for this specific task -- a more careful
  version would tune this per class instead of using one number
  everywhere.
