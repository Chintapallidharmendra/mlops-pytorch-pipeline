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
