"""
Tests for the weightless loader with basic PyTorch models.
"""
import torch.nn as nn
from .utils import check_model_with_and_without_weights


class SimpleModel(nn.Module):
    """A simple model with a few layers."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class NestedModel(nn.Module):
    """A model with nested submodules."""
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def test_simple_model():
    """Test the weightless loader with a simple model."""
    model = SimpleModel()
    model.eval()  # Set to evaluation mode
    input_shape = (1, 3, 32, 32)  # (batch_size, channels, height, width)
    check_model_with_and_without_weights(model, input_shape, simplify=False, opset_version=11)


def test_nested_model():
    """Test the weightless loader with a nested model."""
    model = NestedModel()
    model.eval()  # Set to evaluation mode
    input_shape = (1, 3, 32, 32)  # (batch_size, channels, height, width)
    check_model_with_and_without_weights(model, input_shape, simplify=False, opset_version=11)