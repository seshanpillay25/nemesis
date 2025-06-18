"""
Test Configuration and Fixtures 🏛️

"The sacred testing grounds where models prove their worth"

Pytest configuration and shared fixtures for the Nemesis test suite.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

class SimpleTestModel(nn.Module):
    """Simple test model for consistent testing."""
    
    def __init__(self, input_size: int = 784, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleCNNModel(nn.Module):
    """Simple CNN test model."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@pytest.fixture
def simple_model():
    """Fixture for simple test model."""
    model = SimpleTestModel()
    model.eval()
    return model

@pytest.fixture
def cnn_model():
    """Fixture for CNN test model."""
    model = SimpleCNNModel()
    model.eval()
    return model

@pytest.fixture
def mnist_input():
    """Fixture for MNIST-style input."""
    return torch.randn(1, 1, 28, 28, requires_grad=True)

@pytest.fixture
def cifar_input():
    """Fixture for CIFAR-style input."""
    return torch.randn(1, 3, 32, 32, requires_grad=True)

@pytest.fixture
def flat_input():
    """Fixture for flattened input."""
    return torch.randn(1, 784, requires_grad=True)

@pytest.fixture
def sample_labels():
    """Fixture for sample labels."""
    return torch.randint(0, 10, (5,))

@pytest.fixture
def dummy_dataset():
    """Fixture for dummy dataset."""
    dataset = []
    for i in range(10):
        x = torch.randn(1, 3, 32, 32)
        y = torch.randint(0, 10, (1,)).item()
        dataset.append((x, y))
    return dataset

@pytest.fixture
def training_dataset():
    """Fixture for training dataset."""
    dataset = []
    for i in range(20):
        x = torch.randn(1, 784)
        y = torch.randint(0, 10, (1,)).item()
        dataset.append((x, y))
    return dataset

@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    """Fixture for temporary directory."""
    return tmp_path_factory.mktemp("nemesis_tests")

@pytest.fixture
def examples_dir():
    """Fixture for examples directory."""
    import os
    from pathlib import Path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    examples_path = os.path.join(os.path.dirname(current_dir), "examples")
    return Path(examples_path)

# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

# Disable gradient computation for faster tests
@pytest.fixture(autouse=True)
def setup_torch():
    """Set up torch for testing."""
    torch.set_grad_enabled(True)  # Keep gradients for attack tests
    torch.manual_seed(42)
    np.random.seed(42)