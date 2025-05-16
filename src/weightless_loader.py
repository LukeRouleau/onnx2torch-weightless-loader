"""
`onnx2torch` Weightless Loader Patch

This module provides a patch for the `onnx2torch` library that enables
loading ONNX models without their external weight files by using FakeTensors
(torch meta tensors).
"""

import torch
from contextlib import contextmanager

import onnx
from onnx import external_data_helper
from onnx2torch.onnx_tensor import OnnxTensor
from onnx2torch.utils.dtype import onnx_dtype_to_torch_dtype


def to_torch_weightless(self: OnnxTensor) -> torch.Tensor:
    """Create a fake tensor from ONNX TensorProto with the right shape and dtype"""
    shape = [dim for dim in self._proto.dims]
    dtype = onnx_dtype_to_torch_dtype(self._proto.data_type)
    return torch.empty(shape, dtype=dtype, device="meta")


@contextmanager
def weightless_onnx_loader():
    """
    Context manager that patches onnx2torch to load models without external weights.
    
    When used, this allows loading models with external data references
    even if the weight files are missing, by creating FakeTensors with the
    correct shapes and dtypes instead.
    
    Usage:
        with weightless_onnx_loader():
            model = onnx2torch.convert(onnx_model)
    """
    orig_load_extern = onnx.load_external_data_for_model
    orig_to_torch = OnnxTensor.to_torch
    orig_device = torch.get_default_device()

    try:
        # Apply patches to all relevant places
        onnx.load_external_data_for_model = lambda *args, **kwargs: None
        OnnxTensor.to_torch = to_torch_weightless
        torch.set_default_device("meta")
        yield
    finally:
        # Restore original methods
        onnx.load_external_data_for_model = orig_load_extern
        OnnxTensor.to_torch = orig_to_torch
        torch.set_default_device(orig_device)
