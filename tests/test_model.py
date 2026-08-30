import pytest
import torch

from model import SimpleCNN, get_model


def test_get_model_simple_cnn_returns_simple_cnn_instance():
    model = get_model("simple_cnn", num_classes=10)
    assert isinstance(model, SimpleCNN)


def test_get_model_resnet18_has_correct_output_features():
    model = get_model("resnet18", num_classes=10, pretrained=False)
    assert model.fc.out_features == 10


def test_get_model_resnet18_stem_adapted_for_cifar():
    model = get_model("resnet18", num_classes=10, pretrained=False)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, torch.nn.Identity)


def test_get_model_unknown_architecture_raises_value_error():
    with pytest.raises(ValueError, match="Unknown architecture"):
        get_model("not_a_real_architecture", num_classes=10)


@pytest.mark.parametrize("architecture", ["simple_cnn", "resnet18"])
def test_forward_pass_produces_expected_output_shape(architecture):
    model = get_model(architecture, num_classes=10, pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 10)


def test_simple_cnn_respects_custom_num_classes():
    model = SimpleCNN(num_classes=7)
    dummy_input = torch.randn(1, 3, 32, 32)

    output = model(dummy_input)

    assert output.shape == (1, 7)
