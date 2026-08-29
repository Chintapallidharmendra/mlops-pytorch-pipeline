# mlops-pytorch-pipeline

---

#### This is an academic project from MLOps course in Web Enabled M-Tech course. It is implemented and maintained by Chintapalli Dharmendra (DA25M559).

---

### Project Structure

```Shell
mlops-pytorch-pipeline/
··· README.md
··· .gitignore
··· .github/
· ··· workflows/
· ··· ci.yml
··· src/
· ··· train.py
· ··· model.py
· ··· dataset.py
· ··· serve.py
··· configs/
· ··· training_config.yaml

··· docker/
· ··· Dockerfile.train
· ··· Dockerfile.serve
··· k8s/
· ··· namespace.yaml
· ··· training-job.yaml
· ··· serving-deployment.yaml
· ··· serving-service.yaml
· ··· configmap.yaml
· ··· hpa.yaml
··· requirements/
· ··· train.txt
· ··· serve.txt
··· tests/
··· test_model.py
```

### Training Image

Multi-stage image (`docker/Dockerfile.train`) that trains the CIFAR-10 classifier. Stage 1 (`base`) installs the pinned dependencies from `requirements/train.txt` (`torch==2.5.1`, `torchvision==0.20.1`, `pyyaml==6.0.2`); stage 2 (`training`) copies in `src/` and `configs/` and runs `src/train.py`.

Build:

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
```

Run:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1
```

- `configs/training_config.yaml` is baked into the image as the default config; mount your own over it with `-v $(pwd)/configs:/app/configs` to override it at runtime.
- Training progress is logged as JSON lines to stdout per epoch; the best checkpoint (by validation loss) is written to `/app/checkpoints`.
- The first run needs outbound internet access to download the CIFAR-10 dataset and pretrained ResNet18 weights.

### Serving Image

Two-stage image (`docker/Dockerfile.serve`) that serves the trained classifier over a FastAPI app (`src/serve.py`). Stage 1 (`base`) installs the pinned inference-only dependencies from `requirements/serve.txt` (`torch==2.5.1`, `torchvision==0.20.1`, `fastapi==0.115.6`, `uvicorn[standard]==0.34.0`, `python-multipart==0.0.20`, `pillow==11.1.0`); stage 2 (`serving`) copies in `src/`, creates a non-root user, and runs `uvicorn` against `serve:app` on port 8080.

Build:

```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

Run:

```bash
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1
```

- The checkpoint is mounted at runtime, not baked into the image — same pattern as the training image. `serve.py` defaults to `checkpoints/classifier_v1.pt`, overridable via the `MODEL_CHECKPOINT_PATH` env var.
- `GET /health` reports `{"status": "ok"}` once the model has loaded; the image's `HEALTHCHECK` polls this endpoint.
- `POST /predict` takes a multipart file upload under the field name `file` (not `image`) and returns the predicted CIFAR-10 class with per-class probabilities:

```bash
curl -X POST http://localhost:8080/predict -F "file=@test_image.png"
```

- The container runs as a non-root user (`appuser`, uid 1000).
