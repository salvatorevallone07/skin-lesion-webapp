"""Single-image inference. Uses EXACTLY the same pipeline as training
(denoise -> segment -> extract_features) so the features match the trained
distribution."""
import json
import os
import threading

import cv2 as cv
import numpy as np
import torch

from src.classifier import Classifier
from src.preprocessing import preprocess_denoise
from src.segmentation import segment_lesion
from src.features.extractor import extract_features

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")
NORM_PATH = os.path.join(MODEL_DIR, "normalization.json")
EPS = 1e-8

CLASS_NAMES = {0: "melanocytic_nevus", 1: "melanoma"}
DEFAULT_HIDDEN = [256, 128, 64]
DEFAULT_INPUT_DIM = 35  # matches the current feature set; overridden by normalization.json

_lock = threading.Lock()
_state = {"model": None, "mean": None, "std": None, "feature_names": None,
          "threshold": 0.5, "version": "untrained-dev", "trained": False,
          "input_dim": DEFAULT_INPUT_DIM, "hidden_dims": DEFAULT_HIDDEN, "dropout": 0.3}


def _load_model():
    with _lock:
        if _state["model"] is not None:
            return _state

        if os.path.exists(MODEL_PATH) and os.path.exists(NORM_PATH):
            with open(NORM_PATH) as f:
                norm = json.load(f)
            input_dim = int(norm.get("input_dim", DEFAULT_INPUT_DIM))
            hidden = tuple(norm.get("hidden_dims", DEFAULT_HIDDEN))
            dropout = float(norm.get("dropout", 0.3))
            model = Classifier(input_dim=input_dim, hidden_dims=hidden, dropout=dropout)
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            _state.update({
                "model": model,
                "mean": np.asarray(norm["mean"], dtype=np.float32),
                "std": np.asarray(norm["std"], dtype=np.float32),
                "feature_names": norm.get("feature_names"),
                "threshold": float(norm.get("threshold", 0.5)),
                "version": str(norm.get("version", "trained")),
                "trained": True,
                "input_dim": input_dim,
                "hidden_dims": list(hidden),
                "dropout": dropout,
            })
        else:
            input_dim = _state["input_dim"]
            model = Classifier(input_dim=input_dim, hidden_dims=tuple(_state["hidden_dims"]))
            model.eval()
            _state.update({
                "model": model,
                "mean": np.zeros(input_dim, dtype=np.float32),
                "std": np.ones(input_dim, dtype=np.float32),
                "trained": False,
                "version": "untrained-dev",
            })
        return _state


def predict_from_bytes(image_bytes: bytes) -> dict:
    state = _load_model()

    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv.imdecode(arr, cv.IMREAD_COLOR)  # BGR
    if image is None:
        raise ValueError("Cannot decode image bytes")
    h, w = image.shape[:2]
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image = preprocess_denoise(image)
    mask = segment_lesion(image)
    names, feats = extract_features(image, mask)

    # Untrained-dev safety net: ensure the demo model matches the feature length
    if not state["trained"] and len(feats) != state["input_dim"]:
        with _lock:
            dim = len(feats)
            state["model"] = Classifier(input_dim=dim, hidden_dims=tuple(state["hidden_dims"]))
            state["model"].eval()
            state["mean"] = np.zeros(dim, dtype=np.float32)
            state["std"] = np.ones(dim, dtype=np.float32)
            state["input_dim"] = dim

    feats_norm = (feats - state["mean"]) / (state["std"] + EPS)
    x = torch.tensor(feats_norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(state["model"](x)).item()

    threshold = state["threshold"]
    pred = 1 if prob >= threshold else 0
    return {
        "predicted_class": CLASS_NAMES[pred],
        "probability_melanoma": prob,
        "threshold_used": threshold,
        "model_version": state["version"],
        "model_trained": state["trained"],
        "image_width": int(w),
        "image_height": int(h),
        "features": {n: float(v) for n, v in zip(names, feats)},
    }
