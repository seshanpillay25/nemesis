"""
Model Utilities 🧠🔧

"Understanding the divine architecture of artificial minds"

Utilities for model analysis, framework detection, and
cross-framework compatibility in adversarial testing.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings


@dataclass
class ModelInfo:
    """Information about a model."""
    framework: str
    model_type: str
    input_shape: Optional[Tuple[int, ...]]
    output_shape: Optional[Tuple[int, ...]]
    num_parameters: Optional[int]
    trainable_parameters: Optional[int]
    device: str
    precision: str


class FrameworkDetector:
    """
    Automatic framework detection for ML models.
    
    Detects PyTorch, TensorFlow, scikit-learn, and other frameworks
    to provide appropriate handling strategies.
    """
    
    @staticmethod
    def detect_framework(model: Any) -> str:
        """Detect the ML framework of a model."""
        
        # PyTorch detection
        if hasattr(model, 'parameters') and hasattr(model, 'forward'):
            if 'torch' in str(type(model)):
                return "pytorch"
        
        # TensorFlow/Keras detection
        if hasattr(model, 'predict') and hasattr(model, 'compile'):
            return "tensorflow"
        
        # Scikit-learn detection
        if hasattr(model, 'fit') and hasattr(model, 'predict'):
            if hasattr(model, 'get_params'):
                return "sklearn"
        
        # JAX detection
        if hasattr(model, '__call__') and 'jax' in str(type(model)):
            return "jax"
        
        # Hugging Face detection
        if hasattr(model, 'config') and hasattr(model, 'forward'):
            return "huggingface"
        
        # ONNX detection
        if hasattr(model, 'run') and 'onnx' in str(type(model)).lower():
            return "onnx"
        
        return "unknown"
    
    @staticmethod
    def get_model_device(model: Any) -> str:
        """Get the device a model is on."""
        framework = FrameworkDetector.detect_framework(model)
        
        if framework == "pytorch":
            try:
                return str(next(model.parameters()).device)
            except:
                return "cpu"
        
        elif framework == "tensorflow":
            try:
                import tensorflow as tf
                if hasattr(tf, 'config'):
                    gpus = tf.config.list_physical_devices('GPU')
                    return "gpu" if gpus else "cpu"
                return "cpu"
            except:
                return "cpu"
        
        else:
            return "cpu"


class ModelWrapper:
    """
    Universal model wrapper for cross-framework compatibility.
    
    Provides a consistent interface for models from different frameworks,
    enabling uniform adversarial testing procedures.
    """
    
    def __init__(self, model: Any, framework: Optional[str] = None):
        self.model = model
        self.framework = framework or FrameworkDetector.detect_framework(model)
        self.device = FrameworkDetector.get_model_device(model)
        self._setup_wrapper()
    
    def _setup_wrapper(self):
        """Setup framework-specific wrapper methods."""
        if self.framework == "pytorch":
            self._predict = self._pytorch_predict
            self._get_gradients = self._pytorch_gradients
        elif self.framework == "tensorflow":
            self._predict = self._tensorflow_predict
            self._get_gradients = self._tensorflow_gradients
        elif self.framework == "sklearn":
            self._predict = self._sklearn_predict
            self._get_gradients = self._sklearn_gradients
        else:
            self._predict = self._generic_predict
            self._get_gradients = self._generic_gradients
    
    def predict(self, x: Any) -> Any:
        """Unified prediction interface."""
        return self._predict(x)
    
    def get_gradients(self, x: Any, target: Optional[Any] = None) -> Any:
        """Unified gradient computation interface."""
        return self._get_gradients(x, target)
    
    def _pytorch_predict(self, x):
        """PyTorch prediction."""
        self.model.eval()
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).float()
        
        # Move to model device
        if hasattr(self.model, 'parameters'):
            device = next(self.model.parameters()).device
            x = x.to(device)
        
        with torch.no_grad():
            return self.model(x)
    
    def _pytorch_gradients(self, x, target=None):
        """PyTorch gradient computation."""
        self.model.eval()
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).float()
        
        x.requires_grad_(True)
        
        # Move to model device
        if hasattr(self.model, 'parameters'):
            device = next(self.model.parameters()).device
            x = x.to(device)
        
        output = self.model(x)
        
        if target is None:
            # Use predicted class as target
            target = torch.argmax(output, dim=-1)
        
        # Compute gradient
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        
        return x.grad
    
    def _tensorflow_predict(self, x):
        """TensorFlow prediction."""
        try:
            import tensorflow as tf
            
            if not isinstance(x, tf.Tensor):
                x = tf.convert_to_tensor(x, dtype=tf.float32)
            
            return self.model(x)
        except ImportError:
            warnings.warn("TensorFlow not available")
            return None
    
    def _tensorflow_gradients(self, x, target=None):
        """TensorFlow gradient computation."""
        try:
            import tensorflow as tf
            
            if not isinstance(x, tf.Tensor):
                x = tf.convert_to_tensor(x, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(x)
                output = self.model(x)
                
                if target is None:
                    target = tf.argmax(output, axis=-1)
                
                loss = tf.keras.losses.sparse_categorical_crossentropy(target, output)
            
            return tape.gradient(loss, x)
        except ImportError:
            warnings.warn("TensorFlow not available")
            return None
    
    def _sklearn_predict(self, x):
        """Scikit-learn prediction."""
        if hasattr(x, 'numpy'):
            x = x.numpy()
        elif isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        
        # Reshape if needed
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(x)
        else:
            return self.model.predict(x)
    
    def _sklearn_gradients(self, x, target=None):
        """Scikit-learn gradient approximation."""
        warnings.warn("Gradient computation not directly available for sklearn models. Using finite differences.")
        
        # Use finite differences for gradient approximation
        if hasattr(x, 'numpy'):
            x = x.numpy()
        elif isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        
        original_shape = x.shape
        x_flat = x.reshape(x.shape[0], -1)
        
        eps = 1e-5
        gradients = np.zeros_like(x_flat)
        
        base_pred = self._sklearn_predict(x)
        
        for i in range(x_flat.shape[1]):
            x_plus = x_flat.copy()
            x_plus[:, i] += eps
            pred_plus = self._sklearn_predict(x_plus.reshape(original_shape))
            
            x_minus = x_flat.copy()
            x_minus[:, i] -= eps
            pred_minus = self._sklearn_predict(x_minus.reshape(original_shape))
            
            gradients[:, i] = (pred_plus - pred_minus) / (2 * eps)
        
        return gradients.reshape(original_shape)
    
    def _generic_predict(self, x):
        """Generic prediction for unknown frameworks."""
        try:
            if hasattr(self.model, 'predict'):
                return self.model.predict(x)
            elif hasattr(self.model, '__call__'):
                return self.model(x)
            else:
                raise NotImplementedError("Model doesn't have predict or __call__ method")
        except Exception as e:
            warnings.warn(f"Generic prediction failed: {e}")
            return None
    
    def _generic_gradients(self, x, target=None):
        """Generic gradient computation."""
        warnings.warn("Gradient computation not implemented for this framework")
        return None


class ModelAnalyzer:
    """
    Comprehensive model analysis utilities.
    
    Provides insights into model architecture, complexity,
    and potential vulnerabilities.
    """
    
    def __init__(self):
        self.framework_detector = FrameworkDetector()
    
    def analyze_model(self, model: Any) -> ModelInfo:
        """Comprehensive model analysis."""
        
        framework = self.framework_detector.detect_framework(model)
        
        # Get basic info
        model_type = self._get_model_type(model, framework)
        input_shape = self._get_input_shape(model, framework)
        output_shape = self._get_output_shape(model, framework)
        num_params = self._count_parameters(model, framework)
        trainable_params = self._count_trainable_parameters(model, framework)
        device = self.framework_detector.get_model_device(model)
        precision = self._get_precision(model, framework)
        
        return ModelInfo(
            framework=framework,
            model_type=model_type,
            input_shape=input_shape,
            output_shape=output_shape,
            num_parameters=num_params,
            trainable_parameters=trainable_params,
            device=device,
            precision=precision
        )
    
    def _get_model_type(self, model, framework):
        """Determine model type/architecture."""
        model_class = model.__class__.__name__
        
        # Common architectures
        if any(x in model_class.lower() for x in ['resnet', 'vgg', 'alexnet', 'densenet']):
            return "CNN"
        elif any(x in model_class.lower() for x in ['lstm', 'gru', 'rnn']):
            return "RNN"
        elif any(x in model_class.lower() for x in ['transformer', 'bert', 'gpt']):
            return "Transformer"
        elif any(x in model_class.lower() for x in ['linear', 'mlp', 'sequential']):
            return "MLP"
        else:
            return model_class
    
    def _get_input_shape(self, model, framework):
        """Get expected input shape."""
        if framework == "pytorch":
            try:
                # Try to find input shape from first layer
                if hasattr(model, 'modules'):
                    for module in model.modules():
                        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                            if isinstance(module, torch.nn.Conv2d):
                                return (module.in_channels, None, None)  # C, H, W
                            else:
                                return (module.in_features,)
                return None
            except:
                return None
        
        elif framework == "tensorflow":
            try:
                if hasattr(model, 'input_shape'):
                    return model.input_shape[1:]  # Remove batch dimension
                return None
            except:
                return None
        
        else:
            return None
    
    def _get_output_shape(self, model, framework):
        """Get output shape."""
        if framework == "pytorch":
            try:
                # Try to find output shape from last layer
                if hasattr(model, 'modules'):
                    modules_list = list(model.modules())
                    for module in reversed(modules_list):
                        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                            if isinstance(module, torch.nn.Conv2d):
                                return (module.out_channels, None, None)
                            else:
                                return (module.out_features,)
                return None
            except:
                return None
        
        elif framework == "tensorflow":
            try:
                if hasattr(model, 'output_shape'):
                    return model.output_shape[1:]  # Remove batch dimension
                return None
            except:
                return None
        
        else:
            return None
    
    def _count_parameters(self, model, framework):
        """Count total parameters."""
        if framework == "pytorch":
            try:
                return sum(p.numel() for p in model.parameters())
            except:
                return None
        
        elif framework == "tensorflow":
            try:
                return model.count_params()
            except:
                return None
        
        else:
            return None
    
    def _count_trainable_parameters(self, model, framework):
        """Count trainable parameters."""
        if framework == "pytorch":
            try:
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            except:
                return None
        
        elif framework == "tensorflow":
            try:
                trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
                return trainable_count
            except:
                return None
        
        else:
            return None
    
    def _get_precision(self, model, framework):
        """Get model precision."""
        if framework == "pytorch":
            try:
                param = next(model.parameters())
                return str(param.dtype)
            except:
                return "unknown"
        
        elif framework == "tensorflow":
            try:
                return str(model.dtype)
            except:
                return "unknown"
        
        else:
            return "unknown"
    
    def get_vulnerability_indicators(self, model: Any) -> Dict[str, Any]:
        """Analyze potential vulnerability indicators."""
        
        info = self.analyze_model(model)
        indicators = {}
        
        # Architecture-based indicators
        if info.model_type == "MLP":
            indicators['gradient_masking_risk'] = "medium"
        elif info.model_type == "CNN":
            indicators['spatial_attack_risk'] = "high"
        elif info.model_type == "Transformer":
            indicators['attention_attack_risk'] = "high"
        
        # Parameter count indicators
        if info.num_parameters and info.num_parameters > 100_000_000:
            indicators['overparameterization_risk'] = "high"
        elif info.num_parameters and info.num_parameters < 1_000_000:
            indicators['underfitting_risk'] = "medium"
        
        # Precision indicators
        if "float16" in info.precision.lower():
            indicators['precision_attack_risk'] = "medium"
        
        return indicators
    
    def suggest_attack_strategies(self, model: Any) -> List[str]:
        """Suggest appropriate attack strategies based on model analysis."""
        
        info = self.analyze_model(model)
        strategies = []
        
        # Universal strategies
        strategies.extend(["FGSM", "PGD"])
        
        # Architecture-specific strategies
        if info.model_type == "CNN":
            strategies.extend(["C&W", "DeepFool", "Spatial transforms"])
        elif info.model_type == "Transformer":
            strategies.extend(["TextFooler", "BERT-Attack"])
        elif info.model_type == "RNN":
            strategies.extend(["HotFlip", "Sequence attacks"])
        
        # Framework-specific strategies
        if info.framework == "pytorch":
            strategies.append("AutoAttack")
        
        return strategies