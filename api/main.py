"""
Tomato Leaf Health Classification API

Endpoints:
  GET  /health       — liveness + readiness check
  GET  /classes      — list output classes
  POST /predict      — upload image → HEALTHY / UNHEALTHY
  GET  /docs         — auto-generated Swagger UI
"""
import io
from contextlib import asynccontextmanager

import numpy as np
import torch
import torchvision.models as tvm
import torchvision.transforms as T
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.model_loader import get_model, get_scaler
from api.schemas import (ClassesResponse, HealthResponse,
                         PredictionResponse, RejectionResponse)
from api.tomato_guard import TomatoGuard

SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
LABEL_NAMES = {0: "HEALTHY", 1: "UNHEALTHY"}

_guard: TomatoGuard | None = None
_embedder_transform = None
_embedder_model = None


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model, scaler, and tomato guard at startup."""
    global _guard, _embedder_transform, _embedder_model

    print("Loading champion model...")
    get_model()
    get_scaler()

    print("Initializing tomato guard...")
    _guard = TomatoGuard()

    # Build MobileNetV2 embedder for inference-time feature extraction
    p = load_params()
    img_size = p["data"]["image_size"]
    mob = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
    mob.classifier = torch.nn.Identity()
    mob.eval()
    _embedder_model = mob
    _embedder_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print("API ready.")
    yield


app = FastAPI(
    title="Tomato Leaf Health Classifier",
    description=(
        "Upload a tomato leaf image to receive a binary health prediction: "
        "**HEALTHY** or **UNHEALTHY**. Non-tomato images are automatically rejected."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_features(image: Image.Image) -> np.ndarray:
    x = _embedder_transform(image).unsqueeze(0)
    with torch.no_grad():
        feat = _embedder_model(x).squeeze().numpy()
    return feat.reshape(1, -1).astype(np.float32)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        model = get_model()
        loaded = model is not None
        mtype = type(model).__name__ if loaded else None
    except Exception:
        loaded, mtype = False, None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_type=mtype,
    )


@app.get("/classes", response_model=ClassesResponse)
def classes():
    return ClassesResponse(
        classes=["HEALTHY", "UNHEALTHY"],
        label_map={"HEALTHY": 0, "UNHEALTHY": 1},
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={422: {"model": RejectionResponse}},
)
async def predict(file: UploadFile = File(...)):
    # ── Validate content type ────────────────────────────────────────────────
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Accepted: {sorted(SUPPORTED_TYPES)}",
        )

    # ── Decode image ─────────────────────────────────────────────────────────
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")

    # ── Tomato guard ─────────────────────────────────────────────────────────
    is_tomato, similarity = _guard.is_tomato(image)
    if not is_tomato:
        raise HTTPException(
            status_code=422,
            detail={
                "filename": file.filename,
                "rejected": True,
                "reason": (
                    f"Image does not appear to be a tomato leaf "
                    f"(similarity={similarity:.3f}, "
                    f"threshold={_guard.threshold:.3f}). "
                    "Please upload a clear photo of a tomato leaf."
                ),
                "tomato_similarity": round(similarity, 4),
                "threshold_used": _guard.threshold,
            },
        )

    # ── Feature extraction ───────────────────────────────────────────────────
    features = extract_features(image)
    scaler = get_scaler()
    if scaler is not None:
        features = scaler.transform(features)

    # ── Prediction ───────────────────────────────────────────────────────────
    model = get_model()
    pred_int = int(model.predict(features)[0])
    pred_label = LABEL_NAMES[pred_int]
    probas = model.predict_proba(features)[0]
    confidence = float(probas[pred_int])

    return PredictionResponse(
        filename=file.filename,
        prediction=pred_label,
        confidence=round(confidence, 4),
        is_tomato=True,
        tomato_similarity=round(similarity, 4),
        model_used=type(model).__name__,
        message=(
            "Leaf appears healthy. No disease detected."
            if pred_int == 0
            else "Disease or pest infestation detected. Consult an agronomist."
        ),
    )
