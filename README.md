# ONNX2Torch Weightless Loader

A patch for ONNX to PyTorch conversion that enables loading only the graph structure from an ONNX model without requiring the external weight files. It uses FakeTensors (meta device tensors) to represent weight parameters.

## Problem Statement

Large neural network models often distribute their weights as separate "external data" files when exported to ONNX format. This separation can pose challenges for:

1. Model architecture analysis tools that need to understand the structure but don't need actual weights
2. Neural network compiler frontends that convert models to intermediate representations (IR)
3. Development workflows where downloading gigabytes of weights is impractical
4. Compute and memory-constrained environments

The standard ONNX loading process requires both the model definition file (.onnx) AND its external weights files to be present, even if you're only interested in the model structure.

## What This Patch Demonstrates

This project demonstrates the necessary modifications to the ONNX and ONNX2Torch systems to enable:

1. Loading ONNX model architecture without requiring the external weight files
2. Converting this structure-only model to PyTorch's FX Graph IR without weight overhead
3. Preserving all structural information (layers, connections, shapes, dtypes) while using tensor metadata only
4. Making minimal, non-invasive patches to the existing libraries via a context manager

The patch specifically shows how to:
- Override the external data loading functionality in ONNX
- Replace tensor conversions with FakeTensors (meta device) representations
- Handle tensor shape and dtype information without actual data
- Maintain graph structure in the resulting PyTorch model

## Features

- Load ONNX models with external weights references without the weight files
- Preserves model structure and tensor shapes/dtypes
- Works with a variety of model architectures (CNNs, Transformers, etc.)
- Minimal, non-intrusive patch that can be toggled with a context manager

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/onnx2torch-weightless-loader.git
cd onnx2torch-weightless-loader
```

#### Setup Venv
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

The patch is provided as a context manager for ONNX model conversion:

```python
import onnx
import onnx2torch
from src.weightless_loader import weightless_onnx_loader

# Load an ONNX model (even if its external weights file is missing)
onnx_model = onnx.load("model.onnx")

# Convert the ONNX model to PyTorch with our weightless loader patch
with weightless_onnx_loader():
    torch_model = onnx2torch.convert(onnx_model)

# Now you have a PyTorch model with the same structure as the original
# but with all parameters on the 'meta' device

# You can now manipulate the model's structure, analyze it, or convert it to FX IR
# without having to load any weights
```

## How It Works

1. The patch temporarily overrides the external data loading functionality in ONNX
2. When converting tensors from ONNX to PyTorch, it creates empty tensors on the meta device with the correct shape and dtype
3. This allows for analyzing model structure without needing the actual weights
4. The context manager ensures that all original functionality is restored after use

## Limitations

This implementation is intentionally minimal to demonstrate the concept, but has several limitations:

1. **Version Specificity**: This patch is tightly coupled to specific versions of ONNX and onnx2torch. Different versions may have different internal APIs that would require modifications to this code.

2. **ONNX Opset Compatibility**: Different models use different ONNX opset versions, each with their own operator specifications. This patch does not handle all possible opsets or operator variations.

3. **Torch Version Dependency**: Changes in PyTorch's meta tensor implementation or FakeTensor API between versions may break functionality.

4. **Custom Node Converter Registries**: Many onnx2torch implementations use custom node converter registries for specific model architectures, which this patch does not account for.

5. **Partial Graph Loading**: Some complex models may still fail to load if they have custom operators or require specialized initialization beyond simple tensor shapes and dtypes.

Despite these limitations, the core concept demonstrated here is sound: patching the external data loading functionality and using meta tensors to represent parameter shapes/dtypes is the general approach needed to make ONNX load graph-only files without weights.

## Running Tests

```bash
pytest tests/
```

## Use Cases

- Demonstrate how neural network compiler frontends can load ONNX files into PyTorch's `fx.GraphModule` IR without needing large weight files in memory or on disk
- Enable efficient model architecture analysis and manipulation without weight overhead
- Support development and testing of model compilers and optimizers with realistic model structures
- Static graph analysis for optimization, quantization planning, or hardware compatibility checks
- Framework for implementing custom weight initialization or specialized loading strategies