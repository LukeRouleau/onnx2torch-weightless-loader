"""
Tests for the main module.
"""

import sys
import os

# Add the parent directory to the path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import example_function


def test_example_function():
    """Test the example function."""
    assert example_function() == "Hello from ONNX2Torch weightless loader!" 