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
