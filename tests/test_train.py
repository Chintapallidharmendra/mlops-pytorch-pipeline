import math

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from model import get_model
from train import evaluate, load_config, train_one_epoch


def _synthetic_loader(num_samples=8, num_classes=10, batch_size=4):
    inputs = torch.randn(num_samples, 3, 32, 32)
    targets = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)


def test_load_config_parses_yaml_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({"training": {"epochs": 5}}))

    config = load_config(str(config_path))

    assert config["training"]["epochs"] == 5


def test_train_one_epoch_returns_finite_loss_and_valid_accuracy():
    model = get_model("simple_cnn", num_classes=10)
    loader = _synthetic_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    loss, accuracy = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

    assert math.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0


def test_evaluate_does_not_update_model_weights():
    model = get_model("simple_cnn", num_classes=10)
    loader = _synthetic_loader()
    criterion = nn.CrossEntropyLoss()
    params_before = [p.clone() for p in model.parameters()]

    loss, accuracy = evaluate(model, loader, criterion, torch.device("cpu"))

    assert all(torch.equal(before, after) for before, after in zip(params_before, model.parameters()))
    assert math.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0
