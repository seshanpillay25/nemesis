"""
Data Utilities 📊🔧

"Preparing the divine feast of data"

Utilities for data loading, preprocessing, and preparation
for adversarial testing and training.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
import warnings


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    size: int
    input_shape: Tuple[int, ...]
    num_classes: int
    data_type: str
    normalization: Optional[Dict[str, Any]] = None


class DataLoader:
    """
    Universal data loader for different ML frameworks and formats.
    
    Handles PyTorch tensors, NumPy arrays, and various dataset formats
    with consistent interface across frameworks.
    """
    
    def __init__(self, framework: str = "pytorch"):
        self.framework = framework.lower()
        self.supported_frameworks = ["pytorch", "tensorflow", "numpy", "sklearn"]
        
        if self.framework not in self.supported_frameworks:
            raise ValueError(f"Framework {framework} not supported. Choose from {self.supported_frameworks}")
    
    def load_dataset(self, 
                    data: Any,
                    labels: Optional[Any] = None,
                    batch_size: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """Load dataset from various formats."""
        
        if self.framework == "pytorch":
            return self._load_pytorch_dataset(data, labels, batch_size)
        elif self.framework == "tensorflow":
            return self._load_tensorflow_dataset(data, labels, batch_size)
        elif self.framework == "numpy":
            return self._load_numpy_dataset(data, labels, batch_size)
        elif self.framework == "sklearn":
            return self._load_sklearn_dataset(data, labels, batch_size)
        else:
            raise NotImplementedError(f"Loading for {self.framework} not implemented")
    
    def _load_pytorch_dataset(self, data, labels, batch_size) -> List[Tuple[torch.Tensor, int]]:
        """Load PyTorch dataset."""
        if hasattr(data, '__getitem__') and hasattr(data, '__len__'):
            # Assume it's a PyTorch dataset
            dataset = []
            for i in range(len(data)):
                item = data[i]
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    x, y = item
                else:
                    x = item
                    y = labels[i] if labels is not None else 0
                
                # Ensure tensor format
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x)
                if not isinstance(y, (int, torch.Tensor)):
                    y = int(y) if np.isscalar(y) else torch.tensor(y)
                
                dataset.append((x, y))
            
            return dataset[:batch_size] if batch_size else dataset
        else:
            # Assume numpy arrays or lists
            if labels is None:
                raise ValueError("Labels required when data is not a dataset object")
            
            dataset = []
            for i in range(len(data)):
                x = torch.tensor(data[i]) if not isinstance(data[i], torch.Tensor) else data[i]
                y = int(labels[i])
                dataset.append((x, y))
            
            return dataset[:batch_size] if batch_size else dataset
    
    def _load_tensorflow_dataset(self, data, labels, batch_size) -> List[Tuple[Any, Any]]:
        """Load TensorFlow dataset."""
        try:
            import tensorflow as tf
            
            if isinstance(data, tf.data.Dataset):
                dataset = []
                for item in data.take(batch_size or 1000):
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        x, y = item
                    else:
                        x = item
                        y = 0  # Default label
                    dataset.append((x.numpy(), int(y.numpy()) if hasattr(y, 'numpy') else y))
                return dataset
            else:
                # Convert to numpy and use numpy loader
                if hasattr(data, 'numpy'):
                    data = data.numpy()
                if hasattr(labels, 'numpy'):
                    labels = labels.numpy()
                return self._load_numpy_dataset(data, labels, batch_size)
        except ImportError:
            warnings.warn("TensorFlow not available, falling back to numpy")
            return self._load_numpy_dataset(data, labels, batch_size)
    
    def _load_numpy_dataset(self, data, labels, batch_size) -> List[Tuple[np.ndarray, int]]:
        """Load NumPy dataset."""
        if labels is None:
            raise ValueError("Labels required for numpy arrays")
        
        dataset = []
        for i in range(len(data)):
            x = np.array(data[i]) if not isinstance(data[i], np.ndarray) else data[i]
            y = int(labels[i])
            dataset.append((x, y))
        
        return dataset[:batch_size] if batch_size else dataset
    
    def _load_sklearn_dataset(self, data, labels, batch_size) -> List[Tuple[np.ndarray, int]]:
        """Load scikit-learn compatible dataset."""
        return self._load_numpy_dataset(data, labels, batch_size)
    
    def get_dataset_info(self, dataset: List[Tuple[Any, Any]]) -> DatasetInfo:
        """Extract information about the dataset."""
        if not dataset:
            raise ValueError("Empty dataset")
        
        sample_x, sample_y = dataset[0]
        
        # Determine input shape
        if hasattr(sample_x, 'shape'):
            input_shape = sample_x.shape
        else:
            input_shape = (len(sample_x),) if hasattr(sample_x, '__len__') else (1,)
        
        # Determine number of classes
        all_labels = [y for _, y in dataset]
        num_classes = len(set(all_labels))
        
        # Determine data type
        if isinstance(sample_x, torch.Tensor):
            data_type = "pytorch"
        elif isinstance(sample_x, np.ndarray):
            data_type = "numpy"
        else:
            data_type = "unknown"
        
        return DatasetInfo(
            name="custom_dataset",
            size=len(dataset),
            input_shape=input_shape,
            num_classes=num_classes,
            data_type=data_type
        )


class DataPreprocessor:
    """
    Data preprocessing utilities for adversarial robustness testing.
    
    Handles normalization, augmentation, and format conversion.
    """
    
    def __init__(self):
        self.normalizers = {}
    
    def normalize_batch(self, 
                       batch: List[Tuple[Any, Any]],
                       method: str = "standard",
                       statistics: Optional[Dict] = None) -> List[Tuple[Any, Any]]:
        """Normalize a batch of data."""
        
        if method == "standard":
            return self._normalize_standard(batch, statistics)
        elif method == "minmax":
            return self._normalize_minmax(batch, statistics)
        elif method == "imagenet":
            return self._normalize_imagenet(batch)
        elif method == "cifar":
            return self._normalize_cifar(batch)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _normalize_standard(self, batch, statistics):
        """Standard normalization (z-score)."""
        if statistics is None:
            # Calculate statistics from batch
            all_data = [x for x, _ in batch]
            if isinstance(all_data[0], torch.Tensor):
                stacked = torch.stack(all_data)
                mean = stacked.mean(dim=0)
                std = stacked.std(dim=0)
            else:
                stacked = np.stack(all_data)
                mean = stacked.mean(axis=0)
                std = stacked.std(axis=0)
        else:
            mean = statistics['mean']
            std = statistics['std']
        
        normalized_batch = []
        for x, y in batch:
            if isinstance(x, torch.Tensor):
                x_norm = (x - mean) / (std + 1e-8)
            else:
                x_norm = (x - mean) / (std + 1e-8)
            normalized_batch.append((x_norm, y))
        
        return normalized_batch
    
    def _normalize_minmax(self, batch, statistics):
        """Min-max normalization."""
        if statistics is None:
            all_data = [x for x, _ in batch]
            if isinstance(all_data[0], torch.Tensor):
                stacked = torch.stack(all_data)
                min_val = stacked.min()
                max_val = stacked.max()
            else:
                stacked = np.stack(all_data)
                min_val = stacked.min()
                max_val = stacked.max()
        else:
            min_val = statistics['min']
            max_val = statistics['max']
        
        normalized_batch = []
        for x, y in batch:
            x_norm = (x - min_val) / (max_val - min_val + 1e-8)
            normalized_batch.append((x_norm, y))
        
        return normalized_batch
    
    def _normalize_imagenet(self, batch):
        """ImageNet normalization."""
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        normalized_batch = []
        for x, y in batch:
            if isinstance(x, torch.Tensor):
                if x.dim() == 4:  # Batch dimension
                    x_norm = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
                elif x.dim() == 3:  # Single image
                    x_norm = (x - mean.view(3, 1, 1)) / std.view(3, 1, 1)
                else:
                    x_norm = x  # Can't normalize
            else:
                x_norm = x
            normalized_batch.append((x_norm, y))
        
        return normalized_batch
    
    def _normalize_cifar(self, batch):
        """CIFAR normalization."""
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])
        
        normalized_batch = []
        for x, y in batch:
            if isinstance(x, torch.Tensor):
                if x.dim() == 4:
                    x_norm = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
                elif x.dim() == 3:
                    x_norm = (x - mean.view(3, 1, 1)) / std.view(3, 1, 1)
                else:
                    x_norm = x
            else:
                x_norm = x
            normalized_batch.append((x_norm, y))
        
        return normalized_batch
    
    def augment_for_robustness(self, 
                             batch: List[Tuple[Any, Any]],
                             augmentation_types: List[str] = None) -> List[Tuple[Any, Any]]:
        """Apply data augmentation for robustness training."""
        
        if augmentation_types is None:
            augmentation_types = ["noise", "blur"]
        
        augmented_batch = []
        
        for x, y in batch:
            augmented_batch.append((x, y))  # Original
            
            for aug_type in augmentation_types:
                if aug_type == "noise":
                    x_aug = self._add_gaussian_noise(x, std=0.01)
                elif aug_type == "blur":
                    x_aug = self._add_blur(x)
                elif aug_type == "rotation":
                    x_aug = self._add_rotation(x)
                else:
                    continue
                
                augmented_batch.append((x_aug, y))
        
        return augmented_batch
    
    def _add_gaussian_noise(self, x, std=0.01):
        """Add Gaussian noise."""
        if isinstance(x, torch.Tensor):
            noise = torch.randn_like(x) * std
            return x + noise
        else:
            noise = np.random.normal(0, std, x.shape)
            return x + noise
    
    def _add_blur(self, x):
        """Add simple blur effect."""
        # Simplified blur - in practice, use proper image processing
        if isinstance(x, torch.Tensor) and x.dim() >= 3:
            # Simple averaging blur
            kernel_size = 3
            kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
            if x.dim() == 3:
                x_blurred = torch.nn.functional.conv2d(x.unsqueeze(0), kernel, padding=1).squeeze(0)
            else:
                x_blurred = x  # Can't blur easily
            return x_blurred
        else:
            return x  # Can't blur non-image data
    
    def _add_rotation(self, x):
        """Add small rotation."""
        # Placeholder - in practice, use proper geometric transformations
        return x


class InputNormalizer:
    """
    Specialized normalizer for consistent input handling across frameworks.
    """
    
    def __init__(self, target_framework: str = "pytorch"):
        self.target_framework = target_framework.lower()
    
    def normalize_input(self, x: Any) -> Any:
        """Normalize input to target framework format."""
        
        if self.target_framework == "pytorch":
            return self._to_pytorch(x)
        elif self.target_framework == "tensorflow":
            return self._to_tensorflow(x)
        elif self.target_framework == "numpy":
            return self._to_numpy(x)
        else:
            return x
    
    def _to_pytorch(self, x):
        """Convert to PyTorch tensor."""
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        else:
            try:
                return torch.tensor(x).float()
            except:
                return x
    
    def _to_tensorflow(self, x):
        """Convert to TensorFlow tensor."""
        try:
            import tensorflow as tf
            if hasattr(tf, 'convert_to_tensor'):
                return tf.convert_to_tensor(x)
            else:
                return x
        except ImportError:
            return self._to_numpy(x)
    
    def _to_numpy(self, x):
        """Convert to NumPy array."""
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        else:
            try:
                return np.array(x)
            except:
                return x
    
    def ensure_batch_dimension(self, x: Any) -> Any:
        """Ensure input has batch dimension."""
        if isinstance(x, torch.Tensor):
            if x.dim() == 1:
                return x.unsqueeze(0)
            elif x.dim() == 3:  # Add batch dim for images
                return x.unsqueeze(0)
            return x
        elif isinstance(x, np.ndarray):
            if x.ndim == 1:
                return x.reshape(1, -1)
            elif x.ndim == 3:  # Add batch dim for images
                return x.reshape(1, *x.shape)
            return x
        else:
            return x
    
    def remove_batch_dimension(self, x: Any) -> Any:
        """Remove batch dimension if batch size is 1."""
        if isinstance(x, torch.Tensor):
            if x.size(0) == 1:
                return x.squeeze(0)
            return x
        elif isinstance(x, np.ndarray):
            if x.shape[0] == 1:
                return x.squeeze(0)
            return x
        else:
            return x