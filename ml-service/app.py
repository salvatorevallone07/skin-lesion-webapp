"""FastAPI inference microservice.
Endpoints:
  GET  /health   -> liveness + whether a trained model is loaded
  GET  /info     -> model metadata
  POST /predict  -> multipart 'file' image -> classification result
"""
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from inference import predict_from_bytes, _load_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ml-service")

app = FastAPI(title="Skin Lesion Inference Service", version="1.0.0")

ALLOWED = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}


@app.on_event("startup")
def _startup():
    state = _load_model()
    if not state["trained"]:
        log.warning("No trained model found in MODEL_DIR. Running with an UNTRAINED "
                    "model (predictions are NOT meaningful). Run train.py to export "
                    "model.pt + normalization.json.")
    else:
        log.info("Loaded trained model version=%s", state["version"])


@app.get("/health")
def health():
    state = _load_model()
    return {"status": "ok", "model_trained": state["trained"], "model_version": state["version"]}


@app.get("/info")
def info():
    state = _load_model()
    return {
        "model_version": state["version"],
        "model_trained": state["trained"],
        "framework": "pytorch",
        "input_dim": state["input_dim"],
        "threshold": state["threshold"],
        "classes": ["melanocytic_nevus", "melanoma"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        result = predict_from_bytes(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    return JSONResponse(result)
