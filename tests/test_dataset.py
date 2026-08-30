import numpy as np
from PIL import Image
from torchvision import transforms

from dataset import CIFAR10_CLASSES, get_transforms


def test_cifar10_classes_has_ten_unique_entries():
    assert len(CIFAR10_CLASSES) == 10
    assert len(set(CIFAR10_CLASSES)) == 10


def test_train_transforms_include_augmentation():
    transform_types = [type(t) for t in get_transforms(train=True).transforms]
    assert transforms.RandomHorizontalFlip in transform_types
    assert transforms.RandomCrop in transform_types


def test_eval_transforms_exclude_augmentation():
    transform_types = [type(t) for t in get_transforms(train=False).transforms]
    assert transforms.RandomHorizontalFlip not in transform_types
    assert transforms.RandomCrop not in transform_types


def test_eval_transform_produces_expected_tensor_shape():
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))

    tensor = get_transforms(train=False)(image)

    assert tensor.shape == (3, 32, 32)
