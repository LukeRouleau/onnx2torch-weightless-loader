"""
Tests for the weightless loader with torchvision models.
"""

import pytest
import torch
import torchvision.models as models
from .utils import check_model_with_and_without_weights


@pytest.mark.parametrize(
    "model_name,input_shape,simplify,opset_version",
    [
        ("resnet18", (1, 3, 224, 224), True, 13),
        ("alexnet", (1, 3, 224, 224), True, 13),
        ("efficientnet_b0", (1, 3, 224, 224), True, 13),
    ],
)
def test_torchvision_models(model_name, input_shape, simplify, opset_version):
    """Test the weightless loader with various torchvision models."""
    # Get the model
    model_fn = getattr(models, model_name)
    model = model_fn(pretrained=False)  # Use weights=None for newer versions
    model.eval()
    check_model_with_and_without_weights(model, input_shape, simplify, opset_version)
