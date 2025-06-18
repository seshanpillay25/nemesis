"""
Barrier Shield 🚧

"The vigilant guardian that detects adversarial threats"

Implementation of adversarial detection defenses that identify
malicious inputs before they can cause harm.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Dict, Tuple
from ..base import DefenseBase, DefenseResult

class Barrier(DefenseBase):
    """
    Barrier - Adversarial Detection Defense
    
    Like a mystical barrier that reveals hidden threats,
    this defense detects adversarial examples before they strike.
    """
    
    def __init__(self):
        super().__init__(
            name="Barrier",
            description="Adversarial detection that reveals hidden threats before they can strike",
            defense_type="detection"
        )
        self.detector_model = None
        self.detection_threshold = 0.5
        self.calibrated = False
    
    def forge_defense(self, x: Any, detection_method: str = "statistical",
                     threshold: float = 0.5, **kwargs) -> DefenseResult:
        """
        Forge a Barrier defense through adversarial detection.
        
        Args:
            x: Input to analyze for adversarial content
            detection_method: Detection method ("statistical", "neural", "ensemble")
            threshold: Detection threshold
            **kwargs: Additional detection parameters
            
        Returns:
            DefenseResult indicating if input is adversarial
        """
        original_input = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        
        if detection_method == "statistical":
            return self._statistical_detection(x, threshold, original_input)
        elif detection_method == "neural":
            return self._neural_detection(x, threshold, original_input)
        elif detection_method == "ensemble":
            return self._ensemble_detection(x, threshold, original_input)
        elif detection_method == "gradient_analysis":
            return self._gradient_analysis_detection(x, threshold, original_input)
        else:
            raise ValueError(f"Unknown detection method: {detection_method}")
    
    def _statistical_detection(self, x: torch.Tensor, threshold: float,
                              original_input: torch.Tensor) -> DefenseResult:
        """Detect adversarial examples using statistical analysis."""
        
        # Statistical features for detection
        features = self._extract_statistical_features(x)
        
        # Simple threshold-based detection
        anomaly_score = self._compute_anomaly_score(features)
        
        # Determine if input is adversarial
        is_adversarial = anomaly_score > threshold
        
        # If adversarial, apply rejection or purification
        if is_adversarial:
            defended_input = self._reject_or_purify(x)
            defense_applied = True
        else:
            defended_input = x
            defense_applied = False
        
        return DefenseResult(
            original_input=original_input,
            defended_input=defended_input,
            original_prediction=None,
            defended_prediction=None,
            defense_applied=defense_applied,
            confidence_change=0.0,
            defense_strength=anomaly_score,
            defense_name=self.name,
            metadata={
                'detection_method': 'statistical',
                'anomaly_score': anomaly_score,
                'threshold': threshold,
                'is_adversarial': is_adversarial,
                'features': features,
            }
        )
    
    def _neural_detection(self, x: torch.Tensor, threshold: float,
                         original_input: torch.Tensor) -> DefenseResult:
        """Detect adversarial examples using neural network detector."""
        
        if self.detector_model is None:
            # Create simple detector if none exists
            self.detector_model = self._create_simple_detector(x)
        
        # Get detection score
        with torch.no_grad():
            detection_logits = self.detector_model(x)
            detection_prob = torch.sigmoid(detection_logits).item()
        
        # Determine if adversarial
        is_adversarial = detection_prob > threshold
        
        if is_adversarial:
            defended_input = self._reject_or_purify(x)
            defense_applied = True
        else:
            defended_input = x
            defense_applied = False
        
        return DefenseResult(
            original_input=original_input,
            defended_input=defended_input,
            original_prediction=None,
            defended_prediction=None,
            defense_applied=defense_applied,
            confidence_change=0.0,
            defense_strength=detection_prob,
            defense_name=self.name,
            metadata={
                'detection_method': 'neural',
                'detection_probability': detection_prob,
                'threshold': threshold,
                'is_adversarial': is_adversarial,
            }
        )
    
    def _ensemble_detection(self, x: torch.Tensor, threshold: float,
                           original_input: torch.Tensor) -> DefenseResult:
        """Detect adversarial examples using ensemble of methods."""
        
        # Combine multiple detection methods
        statistical_result = self._statistical_detection(x, threshold, original_input)
        
        # If neural detector exists, use it too
        if self.detector_model is not None:
            neural_result = self._neural_detection(x, threshold, original_input)
            ensemble_score = (statistical_result.defense_strength + 
                            neural_result.defense_strength) / 2
        else:
            ensemble_score = statistical_result.defense_strength
        
        # Gradient-based detection
        gradient_score = self._compute_gradient_anomaly(x)
        
        # Final ensemble score
        final_score = (ensemble_score + gradient_score) / 2
        is_adversarial = final_score > threshold
        
        if is_adversarial:
            defended_input = self._reject_or_purify(x)
            defense_applied = True
        else:
            defended_input = x
            defense_applied = False
        
        return DefenseResult(
            original_input=original_input,
            defended_input=defended_input,
            original_prediction=None,
            defended_prediction=None,
            defense_applied=defense_applied,
            confidence_change=0.0,
            defense_strength=final_score,
            defense_name=self.name,
            metadata={
                'detection_method': 'ensemble',
                'ensemble_score': final_score,
                'statistical_score': statistical_result.defense_strength,
                'gradient_score': gradient_score,
                'threshold': threshold,
                'is_adversarial': is_adversarial,
            }
        )
    
    def _gradient_analysis_detection(self, x: torch.Tensor, threshold: float,
                                   original_input: torch.Tensor) -> DefenseResult:
        """Detect adversarial examples through gradient analysis."""
        
        gradient_anomaly = self._compute_gradient_anomaly(x)
        is_adversarial = gradient_anomaly > threshold
        
        if is_adversarial:
            defended_input = self._reject_or_purify(x)
            defense_applied = True
        else:
            defended_input = x
            defense_applied = False
        
        return DefenseResult(
            original_input=original_input,
            defended_input=defended_input,
            original_prediction=None,
            defended_prediction=None,
            defense_applied=defense_applied,
            confidence_change=0.0,
            defense_strength=gradient_anomaly,
            defense_name=self.name,
            metadata={
                'detection_method': 'gradient_analysis',
                'gradient_anomaly_score': gradient_anomaly,
                'threshold': threshold,
                'is_adversarial': is_adversarial,
            }
        )
    
    def _extract_statistical_features(self, x: torch.Tensor) -> Dict[str, float]:
        """Extract statistical features for anomaly detection."""
        
        features = {}
        
        # Convert to numpy for easier computation
        x_np = x.detach().cpu().numpy()
        
        # Basic statistics
        features['mean'] = float(np.mean(x_np))
        features['std'] = float(np.std(x_np))
        features['skewness'] = float(self._compute_skewness(x_np))
        features['kurtosis'] = float(self._compute_kurtosis(x_np))
        
        # Frequency domain features
        if len(x_np.shape) >= 2:
            # Compute FFT magnitude
            fft_mag = np.abs(np.fft.fft2(x_np.reshape(-1, *x_np.shape[-2:])))
            features['fft_mean'] = float(np.mean(fft_mag))
            features['fft_std'] = float(np.std(fft_mag))
        
        # High-frequency content (indicator of adversarial perturbations)
        if len(x.shape) >= 3:
            # Compute high-frequency energy
            # Handle different channel counts by converting to grayscale first
            if len(x.shape) == 4:  # Batch
                batch_size, channels, height, width = x.shape
                if channels == 3:
                    # Convert RGB to grayscale
                    x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
                elif channels == 1:
                    x_gray = x
                else:
                    # Take first channel
                    x_gray = x[:, 0:1]
            else:  # Single image
                if x.shape[0] == 3:
                    # Convert RGB to grayscale
                    x_gray = (0.299 * x[0:1] + 0.587 * x[1:2] + 0.114 * x[2:3]).unsqueeze(0)
                elif x.shape[0] == 1:
                    x_gray = x.unsqueeze(0)
                else:
                    # Take first channel
                    x_gray = x[0:1].unsqueeze(0)
            
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype).view(1, 1, 3, 3)
            
            edges_x = F.conv2d(x_gray, sobel_x, padding=1)
            edges_y = F.conv2d(x_gray, sobel_y, padding=1)
            
            edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
            features['edge_energy'] = float(torch.mean(edge_magnitude))
        
        return features
    
    def _compute_anomaly_score(self, features: Dict[str, float]) -> float:
        """Compute anomaly score from statistical features."""
        
        # Simple heuristic-based scoring
        # High edge energy often indicates adversarial perturbations
        score = 0.0
        
        if 'edge_energy' in features:
            # Normalize edge energy (typical range 0-1)
            edge_score = min(1.0, features['edge_energy'] * 10)
            score += 0.4 * edge_score
        
        if 'std' in features:
            # High standard deviation might indicate noise
            std_score = min(1.0, features['std'] * 5)
            score += 0.3 * std_score
        
        if 'fft_std' in features:
            # High frequency content variance
            fft_score = min(1.0, features['fft_std'] * 2)
            score += 0.3 * fft_score
        
        return score
    
    def _compute_gradient_anomaly(self, x: torch.Tensor) -> float:
        """Compute gradient-based anomaly score."""
        
        if not hasattr(self, 'reference_model') or self.reference_model is None:
            # No reference model, return neutral score
            return 0.5
        
        x.requires_grad_(True)
        
        # Compute gradient
        output = self.reference_model(x)
        loss = torch.sum(output)
        loss.backward()
        
        gradient = x.grad
        gradient_norm = torch.norm(gradient).item()
        
        # Normalize gradient norm (heuristic)
        anomaly_score = min(1.0, gradient_norm / 10.0)
        
        return anomaly_score
    
    def _reject_or_purify(self, x: torch.Tensor) -> torch.Tensor:
        """Reject adversarial input or apply purification."""
        
        # Simple purification: add small amount of noise
        noise = torch.randn_like(x) * 0.01
        purified = torch.clamp(x + noise, 0, 1)
        
        return purified
    
    def _create_simple_detector(self, sample_input: torch.Tensor) -> nn.Module:
        """Create a simple adversarial detector network."""
        
        input_size = sample_input.numel()
        
        detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output single detection score
        )
        
        return detector
    
    def _compute_skewness(self, x: np.ndarray) -> float:
        """Compute skewness of distribution."""
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 3))
    
    def _compute_kurtosis(self, x: np.ndarray) -> float:
        """Compute kurtosis of distribution."""
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 4) - 3)
    
    def calibrate_detector(self, clean_samples: list, adversarial_samples: list):
        """Calibrate the detector on clean and adversarial samples."""
        
        # Extract features from both clean and adversarial samples
        clean_features = []
        adv_features = []
        
        for sample in clean_samples:
            features = self._extract_statistical_features(sample)
            clean_features.append(self._compute_anomaly_score(features))
        
        for sample in adversarial_samples:
            features = self._extract_statistical_features(sample)
            adv_features.append(self._compute_anomaly_score(features))
        
        # Find optimal threshold
        clean_scores = np.array(clean_features)
        adv_scores = np.array(adv_features)
        
        # Use median of adversarial scores as threshold
        self.detection_threshold = float(np.median(adv_scores))
        self.calibrated = True
        
        # Compute accuracy
        clean_correct = np.sum(clean_scores < self.detection_threshold)
        adv_correct = np.sum(adv_scores >= self.detection_threshold)
        
        accuracy = (clean_correct + adv_correct) / (len(clean_scores) + len(adv_scores))
        
        return {
            'threshold': self.detection_threshold,
            'accuracy': accuracy,
            'clean_accuracy': clean_correct / len(clean_scores),
            'adversarial_detection_rate': adv_correct / len(adv_scores)
        }
    
    def set_reference_model(self, model: Any):
        """Set reference model for gradient-based detection."""
        self.reference_model = model