"""
main.py -- FastAPI app. Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Two endpoints, matching the brief exactly:
  POST /detect  -- Part A: image in, boxes/classes/confidences out
  POST /ask     -- Part B: image + natural-language question in, reasoned answer out

Includes logging + error handling (bonus criterion: "logging, error handling").
"""
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from api.detector import detect, get_model
from api.reasoning import answer_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ppe-api")

app = FastAPI(title="Constrained PPE Detection & Reasoning API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


@app.on_event("startup")
async def load_model_on_startup():
    # Load the model once at startup rather than on first request, so the
    # first real request isn't slowed down and so we fail fast if weights
    # are missing/corrupt instead of failing silently mid-request.
    try:
        get_model()
        logger.info("RT-DETR model loaded successfully.")
    except Exception:
        logger.exception("Failed to load model weights at startup. Check models/best.pt exists.")


def _validate_and_save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        size = 0
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Image too large (max 15MB).")
            tmp.write(chunk)
        return tmp.name


@app.post("/detect")
async def detect_endpoint(image: UploadFile = File(...), conf_threshold: float = 0.25):
    """
    Sample request:  curl -X POST -F "image=@site1.jpg" -F "conf_threshold=0.3" http://localhost:8000/detect

    Sample response:
    {
      "detections": [
        {"class": "Hardhat", "confidence": 0.91, "box": [120.0, 45.0, 210.0, 160.0]},
        {"class": "NO-Safety Vest", "confidence": 0.67, "box": [300.0, 80.0, 400.0, 260.0]}
      ]
    }
    """
    if not (0.0 <= conf_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="conf_threshold must be between 0 and 1.")

    temp_path = _validate_and_save_upload(image)
    logger.info(f"/detect called: file={image.filename}, conf_threshold={conf_threshold}")
    try:
        detections = detect(temp_path, conf_threshold=conf_threshold)
        logger.info(f"/detect returned {len(detections)} detections")
        return {"detections": detections}
    except FileNotFoundError:
        logger.exception("Model weights not found.")
        raise HTTPException(status_code=503, detail="Model weights not found on server. Has training completed and models/best.pt been placed?")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error during detection.")
        raise HTTPException(status_code=500, detail="Internal error during detection.")
    finally:
        Path(temp_path).unlink(missing_ok=True)


@app.post("/ask")
async def ask_endpoint(image: UploadFile = File(...), question: str = Form(...)):
    """
    Sample request:
      curl -X POST -F "image=@site1.jpg" -F "question=Is anyone not wearing a helmet?" \
           http://localhost:8000/ask

    Sample response (confident):
    {
      "used_detector": true,
      "answer": "Yes -- detected violation(s): NO-Hardhat (1 instance(s)).",
      "confident": true,
      "raw_detections": [...]
    }

    Sample response (guardrail firing):
    {
      "used_detector": true,
      "answer": "I don't have enough confident detection evidence to answer this reliably. (No sufficiently confident detections in this image.)",
      "confident": false,
      "raw_detections": [...]
    }
    """
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    temp_path = _validate_and_save_upload(image)
    logger.info(f"/ask called: file={image.filename}, question='{question}'")
    try:
        result = answer_question(question, temp_path)
        logger.info(f"/ask -> confident={result['confident']}, used_detector={result['used_detector']}")
        return result
    except FileNotFoundError:
        logger.exception("Model weights not found.")
        raise HTTPException(status_code=503, detail="Model weights not found on server.")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error during reasoning.")
        raise HTTPException(status_code=500, detail="Internal error during reasoning.")
    finally:
        Path(temp_path).unlink(missing_ok=True)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception(f"Unhandled exception on {request.url.path}")
    return JSONResponse(status_code=500, content={"detail": "Unexpected server error."})


@app.get("/health")
async def health():
    return {"status": "ok"}
