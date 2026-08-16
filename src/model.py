import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def _build_resnet18(num_classes: int, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    net = resnet18(weights=weights)

    # Adapt the ImageNet stem for 32x32 CIFAR input: the stock 7x7/stride-2 conv
    # + stride-2 maxpool collapses a 32x32 image almost to nothing before the
    # first residual block. conv1 is reinitialized from scratch here even when
    # pretrained=True, so only blocks 2+ retain pretrained ImageNet weights.
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, num_classes)

    return net


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


def get_model(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if architecture == "resnet18":
        return _build_resnet18(num_classes=num_classes, pretrained=pretrained)
    if architecture == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {architecture!r}. Supported: 'resnet18', 'simple_cnn'")
