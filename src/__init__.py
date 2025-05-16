"""
ONNX2Torch Weightless Loader Patch

This package provides a patch for onnx2torch to enable loading ONNX models
without their external weight files.
"""

from .weightless_loader import weightless_onnx_loader

__all__ = ["weightless_onnx_loader"] 