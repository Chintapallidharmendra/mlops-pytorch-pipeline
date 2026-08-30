import importlib
import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from dataset import CIFAR10_CLASSES
from model import get_model


@pytest.fixture()
def checkpoint_path(tmp_path):
    model = get_model("simple_cnn", num_classes=10)
    path = tmp_path / "test_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "simple_cnn",
            "num_classes": 10,
        },
        path,
    )
    return path


@pytest.fixture()
def client(checkpoint_path, monkeypatch):
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(checkpoint_path))
    import serve

    importlib.reload(serve)
    with TestClient(serve.app) as test_client:
        yield test_client


def test_health_endpoint_reports_ok_once_model_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_valid_prediction(client):
    image = Image.new("RGB", (32, 32), color=(128, 64, 32))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post("/predict", files={"file": ("test.png", buffer, "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in CIFAR10_CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 10


def test_predict_endpoint_rejects_non_image_upload(client):
    response = client.post(
        "/predict", files={"file": ("not_an_image.txt", b"not an image", "text/plain")}
    )

    assert response.status_code == 400
