"""
Test utilities for the ONNX2Torch Weightless Loader.
"""

import os
import tempfile
import torch
import onnx
import onnx2torch
import onnxsim
from src.weightless_loader import weightless_onnx_loader


def export_model_with_external_weights(model, input_shape, simplify, opset_version, onnx_path=None):
    """
    Export a PyTorch model to ONNX with external weights.
    
    Args:
        model: PyTorch model to export
        input_shape: Shape of input tensor for tracing
        onnx_path: Path to save the ONNX model (default: temporary file)
        
    Returns:
        tuple: (onnx_path, weights_path) paths to the model and its weights
    """
    if onnx_path is None:
        temp_dir = tempfile.mkdtemp()
        onnx_path = os.path.join(temp_dir, "model.onnx")
    weights_path = f"{onnx_path}.data"
    
    # Export the model with external data
    torch.onnx.export(
        model,
        torch.randn(input_shape),
        onnx_path,
        dynamo=True,  # required for external data
        external_data=True,
        keep_initializers_as_inputs=False,
    )

    # Simplify the model
    onnx_model = onnx.load(onnx_path)
    if simplify:
        onnx_model, _ = onnxsim.simplify(onnx_model)
    onnx_model.opset_import[0].version = opset_version
    onnx_model = onnx.version_converter.convert_version(onnx_model, opset_version)
    onnx.save(onnx_model, onnx_path)

    # Assert that there are external weights alongside the onnx model
    assert os.path.exists(onnx_path)
    assert os.path.exists(weights_path), "External weights file was not created"
    return onnx_path, weights_path


def check_model_with_and_without_weights(model, input_shape, simplify, opset_version):
    """
    Test a model by exporting to ONNX with external weights,
    then loading with and without the weights file.
    
    Args:
        model: PyTorch model to test
        input_shape: Shape of input tensor
        
    Returns:
        dict: Results of the test including models and paths
    """
    # Export the model
    onnx_path, weights_path = export_model_with_external_weights(model, input_shape, simplify, opset_version)

    # First, verify we can load the model with weights
    onnx.load(onnx_path)
    torch_model_with_weights = onnx2torch.convert(onnx_path)
    
    # Now delete the weights file
    os.remove(weights_path)
    
    # Try loading without weights
    with weightless_onnx_loader():
        torch_model_no_weights = onnx2torch.convert(onnx_path)
    
    # Verify the model structure
    original_param_count = sum(1 for _ in torch_model_with_weights.parameters())
    weightless_param_count = sum(1 for _ in torch_model_no_weights.parameters())
    assert weightless_param_count == original_param_count, \
        f"Parameter count mismatch: {weightless_param_count} vs {original_param_count}"
    
    # Verify the model structure
    original_nodes = [node for node in torch_model_with_weights.graph.nodes]
    weightless_nodes = [node for node in torch_model_no_weights.graph.nodes]
    assert len(original_nodes) == len(weightless_nodes), \
        f"Node count mismatch: {len(original_nodes)} vs {len(weightless_nodes)}"
    
    original_node_names = [node.name for node in original_nodes]
    weightless_node_names = [node.name for node in weightless_nodes]
    assert original_node_names == weightless_node_names, \
        f"Node name mismatch: {original_node_names} vs {weightless_node_names}"

    # Cleanup
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    
    return {
        "original_model": model,
        "torch_model_with_weights": torch_model_with_weights,
        "torch_model_weightless": torch_model_no_weights,
        "onnx_path": onnx_path,
        "weights_path": weights_path,
    } 