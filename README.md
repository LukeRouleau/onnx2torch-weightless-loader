# ONNX2Torch Weightless Loader

A patch for ONNX to PyTorch conversion that enables loading only the graph structure from an ONNX model without requiring the external weight files. It uses FakeTensors (meta device tensors) to represent weight parameters.

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
```

## How It Works

1. The patch temporarily overrides the external data loading functionality in ONNX
2. When converting tensors from ONNX to PyTorch, it creates empty tensors on the meta device with the correct shape and dtype
3. This allows for analyzing model structure without needing the actual weights
4. The context manager ensures that all original functionality is restored after use

## Running Tests

```bash
pytest tests/
```

## Use Cases

- Demonstrate how NN compiler frontends can load ONNX files for large models into `fx.GraphModule` IR without needing large weight files in mem or on disk.
- Model architecture analysis without downloading large weight files
- Static graph analysis where weights aren't needed
- Framework for implementing weight initialization or specialized loading strategies
