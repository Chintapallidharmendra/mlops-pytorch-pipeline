import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import transforms

from dataset import CIFAR10_CLASSES, get_transforms
from model import get_model

MODEL_CHECKPOINT_PATH = os.environ.get(
    "MODEL_CHECKPOINT_PATH", "checkpoints/classifier_v1.pt"
)

_state: dict = {"model": None, "device": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        MODEL_CHECKPOINT_PATH, map_location=device, weights_only=True
    )

    model = get_model(
        architecture=checkpoint["architecture"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    _state["model"] = model
    _state["device"] = device
    yield
    _state.clear()


app = FastAPI(lifespan=lifespan)

_predict_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    get_transforms(train=False),
])


class HealthResponse(BaseModel):
    status: str


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health", response_model=HealthResponse)
async def health():
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400, detail="uploaded file is not a valid image"
        )

    tensor = _predict_transform(image).unsqueeze(0).to(_state["device"])

    with torch.no_grad():
        logits = _state["model"](tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_idx = int(torch.argmax(probs).item())
    return PredictResponse(
        predicted_class=CIFAR10_CLASSES[top_idx],
        confidence=round(float(probs[top_idx]), 4),
        probabilities={
            cls: round(float(p), 4) for cls, p in zip(CIFAR10_CLASSES, probs)
        },
    )
