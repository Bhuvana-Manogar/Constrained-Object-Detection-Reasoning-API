"""
reasoning.py -- Part B: the "minimal reasoning layer."

IMPORTANT (defend this explicitly): this file contains NO agentic framework.
No LangChain, no CrewAI, no AutoGen. It is plain Python control flow: a
function that decides whether to call the detector, then a function that
reasons over the returned list-of-dicts, then a function that decides
whether the answer is confident enough to give. That's the whole "decision
layer" the brief asks for -- three plain functions, called in sequence by
answer_question() at the bottom.

If asked "why no framework" in the verbal round: frameworks like LangChain
exist to orchestrate multi-step chains and multiple tools/agents. Here there
is exactly one tool (the detector) and one decision point (need it or not),
so a framework would add abstraction and dependency weight without solving
a problem that actually exists at this scale -- and the brief explicitly
prohibits it anyway.
"""
import re
from typing import List, Dict, Tuple

from api.detector import detect

# --- 1. INTENT ROUTING -------------------------------------------------
# Plain keyword/heuristic classifier. This is intentionally simple and
# auditable rather than an LLM call, because the brief wants to see that
# YOU understand the decision boundary, not that you can prompt a model
# to guess it. Be ready to explain: this will misroute unusual phrasings
# (e.g. sarcasm, indirect questions) -- that's a known, stated limitation,
# not something to hide in the demo.
NO_DETECTION_NEEDED_PATTERNS = [
    r"\bwhat('s| is) your name\b",
    r"\bwho (built|made|created) you\b",
    r"\bwhat can you do\b",
    r"\bhello\b|\bhi\b",
    r"\bwhat is (a|an) (hardhat|helmet|safety vest|ppe)\b",  # definitional, not about THIS image
]

DETECTION_KEYWORDS = [
    "how many", "count", "is anyone", "is there", "are there",
    "wearing", "not wearing", "most common", "what objects",
    "what's in", "what is in", "detect", "helmet", "hardhat", "vest", "mask",
]


def needs_detection(question: str) -> bool:
    q = question.lower().strip()
    for pattern in NO_DETECTION_NEEDED_PATTERNS:
        if re.search(pattern, q):
            return False
    return any(keyword in q for keyword in DETECTION_KEYWORDS)


# --- 2. STRUCTURED REASONING --------------------------------------------
# Pure Python over the detector's structured output. No LLM call needed for
# the actual logic -- an LLM is only used (optionally, in main.py) to phrase
# the final sentence, never to decide the answer's content. This matters:
# if the LLM phrasing step were doing the counting/reasoning itself, you'd
# have no guarantee of correctness and no way to defend a wrong answer.

MIN_CONFIDENT_DETECTIONS = 1     # need at least this many boxes to say anything
LOW_CONFIDENCE_THRESHOLD = 0.35  # boxes below this are too uncertain to trust


def reason_over_detections(question: str, detections: List[Dict]) -> Tuple[str, bool]:
    """
    Returns (answer_text, is_confident).
    is_confident=False means the caller should trigger the guardrail message
    instead of using answer_text as a final answer.
    """
    q = question.lower()

    trustworthy = [d for d in detections if d["confidence"] >= LOW_CONFIDENCE_THRESHOLD]

    if len(trustworthy) < MIN_CONFIDENT_DETECTIONS:
        return "No sufficiently confident detections in this image.", False

    # "how many X" / "count X"
    count_match = re.search(r"how many (\w[\w\s]*?)s?\b", q) or re.search(r"count (\w[\w\s]*)", q)
    if count_match:
        target = count_match.group(1).strip().rstrip("s")
        matches = [d for d in trustworthy if target in d["class"].lower()]
        if not matches and "people" in q or "person" in q or "worker" in q:
            matches = [d for d in trustworthy if d["class"] == "Person"]
        return f"Found {len(matches)} instance(s) matching '{target}'.", True

    # "is anyone not wearing a helmet/hardhat" -- the brief's own example question
    if "not wearing" in q or ("is anyone" in q and ("without" in q or "no " in q)):
        violation_classes = [c for c in ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"]
                              if c.lower().replace("no-", "no ") in q or c.lower() in q
                              or c.split("-")[1].lower() in q]
        if not violation_classes:
            violation_classes = ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"]
        found = [d for d in trustworthy if d["class"] in violation_classes]
        if found:
            classes_found = sorted(set(d["class"] for d in found))
            return f"Yes -- detected violation(s): {', '.join(classes_found)} ({len(found)} instance(s)).", True
        person_count = len([d for d in trustworthy if d["class"] == "Person"])
        if person_count == 0:
            return "No people detected in the image with sufficient confidence to answer.", False
        return "No PPE violations detected among the people identified in this image.", True

    # "most common object"
    if "most common" in q:
        from collections import Counter
        counts = Counter(d["class"] for d in trustworthy)
        top_class, top_count = counts.most_common(1)[0]
        return f"The most common detected object is '{top_class}' ({top_count} instance(s)).", True

    # Fallback: describe what was found, but flag lower confidence since we
    # don't have a specific reasoning rule matched to this phrasing.
    from collections import Counter
    counts = Counter(d["class"] for d in trustworthy)
    summary = ", ".join(f"{v} {k}" for k, v in counts.items())
    return f"Detected: {summary}. (No specific rule matched this question's phrasing -- treat this as a raw summary, not a targeted answer.)", False


# --- 3. CONFIDENCE GUARDRAIL --------------------------------------------
def answer_question(question: str, image_path: str) -> Dict:
    if not needs_detection(question):
        return {
            "used_detector": False,
            "answer": "This question doesn't require analyzing the image content.",
            "confident": True,
        }

    detections = detect(image_path)
    answer_text, is_confident = reason_over_detections(question, detections)

    if not is_confident:
        return {
            "used_detector": True,
            "raw_detections": detections,
            "answer": f"I don't have enough confident detection evidence to answer this reliably. ({answer_text})",
            "confident": False,
        }

    return {
        "used_detector": True,
        "raw_detections": detections,
        "answer": answer_text,
        "confident": True,
    }
