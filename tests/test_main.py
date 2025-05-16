"""
Tests for the weightless loader core functionality.
"""

import sys
import os
import pytest
import torch
import onnx
from onnx.helper import make_tensor
from onnx import TensorProto

# Add the parent directory to the path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.weightless_loader import weightless_onnx_loader, to_torch_weightless
from onnx2torch.onnx_tensor import OnnxTensor


def create_test_onnx_tensor(shape, data_type=TensorProto.FLOAT, with_external_data=False):
    """Create a test ONNX tensor with or without external data reference."""
    tensor = make_tensor(
        name="test_tensor",
        data_type=data_type,
        dims=shape,
        vals=[0.0] * (shape[0] * shape[1]),
    )
    
    if with_external_data:
        tensor.data_location = TensorProto.EXTERNAL
        tensor.external_data.extend([
            onnx.StringStringEntryProto(key="location", value="weights.bin"),
            onnx.StringStringEntryProto(key="offset", value="0"),
            onnx.StringStringEntryProto(key="length", value=str(shape[0] * shape[1] * 4)),
        ])
    
    return tensor


def test_weightless_to_torch_function():
    """Test the to_torch_weightless function directly."""
    # Create a test tensor with external data
    shape = [2, 3]
    proto = create_test_onnx_tensor(shape, with_external_data=True)
    onnx_tensor = OnnxTensor(proto)
    
    # Apply the weightless conversion
    tensor = to_torch_weightless(onnx_tensor)
    
    # Verify the result
    assert tensor.shape == torch.Size(shape)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "meta"


def test_weightless_onnx_loader_context():
    """Test the weightless_onnx_loader context manager."""
    # Create a test tensor with external data
    shape = [2, 3]
    proto = create_test_onnx_tensor(shape, with_external_data=True)
    onnx_tensor = OnnxTensor(proto)
    
    # Original behavior would fail without the external file
    with pytest.raises(Exception):
        # This should fail because external data can't be loaded
        try:
            original_tensor = onnx_tensor.to_torch()
            assert False, "This should have raised an exception"
        except Exception as e:
            raise e
    
    # Test with our context manager
    with weightless_onnx_loader():
        # Now it should work
        tensor = onnx_tensor.to_torch()
        
        # Verify the result
        assert tensor.shape == torch.Size(shape)
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "meta"
