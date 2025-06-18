"""
Resilience Armor 🏰

"Like a fortress that withstands all storms"

Implementation of defensive distillation and other resilience-building
techniques that make models naturally resistant to adversarial attacks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Any, Optional, Dict, List, Tuple
from ..base import DefenseBase, DefenseResult

class Resilience(DefenseBase):
    """
    Resilience - Defensive Distillation and Stability Training
    
    Like an ancient fortress built to withstand sieges,
    Resilience creates naturally robust models through distillation.
    """
    
    def __init__(self, model: Any):
        super().__init__(
            name="Resilience",
            description="Defensive distillation that builds natural resistance like an ancient fortress",
            defense_type="training"
        )
        self.model = model
        self.teacher_model = None
        self.temperature = 20.0
    
    def forge_defense(self, training_data: List[Tuple[Any, Any]],
                     method: str = "distillation", temperature: float = 20.0,
                     training_epochs: int = 15, **kwargs) -> DefenseResult:
        """
        Forge Resilience defense through defensive distillation.
        
        Args:
            training_data: Training dataset
            method: Defense method ("distillation", "label_smoothing", "mixup")
            temperature: Temperature for distillation
            training_epochs: Number of training epochs
            **kwargs: Additional parameters
            
        Returns:
            DefenseResult containing the resilient model
        """
        
        original_model = self._clone_model(self.model)
        
        if method == "distillation":
            resilient_model = self._defensive_distillation(
                training_data, temperature, training_epochs, **kwargs
            )
        elif method == "label_smoothing":
            resilient_model = self._label_smoothing_training(
                training_data, training_epochs, **kwargs
            )
        elif method == "mixup":
            resilient_model = self._mixup_training(
                training_data, training_epochs, **kwargs
            )
        else:
            raise ValueError(f"Unknown resilience method: {method}")
        
        # Evaluate robustness improvement
        improvement = self._evaluate_resilience_improvement(
            original_model, resilient_model, training_data
        )
        
        defense_applied = improvement > 0.03
        
        return DefenseResult(
            original_input=original_model,
            defended_input=resilient_model,
            original_prediction=None,
            defended_prediction=None,
            defense_applied=defense_applied,
            confidence_change=0.0,
            defense_strength=improvement,
            defense_name=self.name,
            metadata={
                'method': method,
                'temperature': temperature,
                'training_epochs': training_epochs,
                'resilience_improvement': improvement,
                'training_samples': len(training_data),
            }
        )
    
    def _defensive_distillation(self, training_data: List[Tuple[Any, Any]],
                               temperature: float, training_epochs: int,
                               **kwargs) -> Any:
        """Implement defensive distillation training."""
        
        self.temperature = temperature
        
        # Phase 1: Train teacher model with high temperature
        print("Phase 1: Training teacher model...")
        self.teacher_model = self._clone_model(self.model)
        self._train_teacher_model(training_data, temperature, training_epochs // 2)
        
        # Phase 2: Train student model (self.model) using teacher's soft targets
        print("Phase 2: Training student model with distillation...")
        self._train_student_model(training_data, temperature, training_epochs // 2)
        
        return self.model
    
    def _train_teacher_model(self, training_data: List[Tuple[Any, Any]],
                           temperature: float, epochs: int):
        """Train the teacher model with high temperature."""
        
        optimizer = optim.Adam(self.teacher_model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for x, y in training_data:
                optimizer.zero_grad()
                
                # Forward pass with temperature
                logits = self.teacher_model(x)
                soft_targets = F.softmax(logits / temperature, dim=-1)
                
                # Loss with temperature scaling
                if isinstance(y, torch.Tensor):
                    loss = F.cross_entropy(logits / temperature, y)
                else:
                    y_tensor = torch.tensor([y], dtype=torch.long)
                    loss = F.cross_entropy(logits / temperature, y_tensor)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"Teacher Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def _train_student_model(self, training_data: List[Tuple[Any, Any]],
                           temperature: float, epochs: int):
        """Train the student model using teacher's soft targets."""
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for x, y in training_data:
                optimizer.zero_grad()
                
                # Get teacher's soft targets
                with torch.no_grad():
                    teacher_logits = self.teacher_model(x)
                    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
                
                # Student forward pass
                student_logits = self.model(x)
                student_soft = F.log_softmax(student_logits / temperature, dim=-1)
                
                # Distillation loss
                distillation_loss = F.kl_div(
                    student_soft, soft_targets, reduction='batchmean'
                ) * (temperature ** 2)
                
                # Hard target loss (small weight)
                if isinstance(y, torch.Tensor):
                    hard_loss = F.cross_entropy(student_logits, y)
                else:
                    y_tensor = torch.tensor([y], dtype=torch.long)
                    hard_loss = F.cross_entropy(student_logits, y_tensor)
                
                # Combined loss
                total_loss = 0.9 * distillation_loss + 0.1 * hard_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"Student Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def _label_smoothing_training(self, training_data: List[Tuple[Any, Any]],
                                 epochs: int, smoothing: float = 0.1) -> Any:
        """Train with label smoothing for increased resilience."""
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for x, y in training_data:
                optimizer.zero_grad()
                
                logits = self.model(x)
                
                # Apply label smoothing
                if isinstance(y, torch.Tensor):
                    num_classes = logits.shape[-1]
                    smooth_targets = self._smooth_labels(y, num_classes, smoothing)
                    loss = F.cross_entropy(logits, smooth_targets)
                else:
                    # For single labels, create smooth distribution
                    num_classes = logits.shape[-1]
                    smooth_targets = torch.full((num_classes,), smoothing / (num_classes - 1))
                    smooth_targets[y] = 1.0 - smoothing
                    loss = F.cross_entropy(logits, smooth_targets.unsqueeze(0))
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"Label Smoothing Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self.model
    
    def _mixup_training(self, training_data: List[Tuple[Any, Any]],
                       epochs: int, alpha: float = 1.0) -> Any:
        """Train with Mixup for improved resilience."""
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Shuffle data for mixup
            import random
            shuffled_data = training_data.copy()
            random.shuffle(shuffled_data)
            
            for i, (x, y) in enumerate(training_data):
                optimizer.zero_grad()
                
                # Get another sample for mixup
                x2, y2 = shuffled_data[i % len(shuffled_data)]
                
                # Mixup
                lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
                mixed_x = lam * x + (1 - lam) * x2
                
                logits = self.model(mixed_x)
                
                # Mixed loss
                if isinstance(y, torch.Tensor) and isinstance(y2, torch.Tensor):
                    loss = lam * F.cross_entropy(logits, y) + (1 - lam) * F.cross_entropy(logits, y2)
                else:
                    y_tensor = torch.tensor([y], dtype=torch.long) if not isinstance(y, torch.Tensor) else y
                    y2_tensor = torch.tensor([y2], dtype=torch.long) if not isinstance(y2, torch.Tensor) else y2
                    loss = lam * F.cross_entropy(logits, y_tensor) + (1 - lam) * F.cross_entropy(logits, y2_tensor)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"Mixup Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self.model
    
    def _smooth_labels(self, labels: torch.Tensor, num_classes: int, 
                      smoothing: float) -> torch.Tensor:
        """Apply label smoothing to hard labels."""
        
        smooth_labels = torch.full((labels.shape[0], num_classes), smoothing / (num_classes - 1))
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
        
        return smooth_labels
    
    def _clone_model(self, model: Any) -> Any:
        """Create a copy of the model."""
        try:
            import copy
            return copy.deepcopy(model)
        except:
            return model
    
    def _evaluate_resilience_improvement(self, original_model: Any,
                                       resilient_model: Any,
                                       test_data: List[Tuple[Any, Any]]) -> float:
        """Evaluate improvement in model resilience."""
        
        if len(test_data) == 0:
            return 0.0
        
        # Test with noisy inputs (simple resilience test)
        test_samples = test_data[:min(10, len(test_data))]
        
        original_robust_acc = 0.0
        resilient_robust_acc = 0.0
        
        for x, y in test_samples:
            # Add noise to test resilience
            noise = torch.randn_like(x) * 0.05
            x_noisy = torch.clamp(x + noise, 0, 1)
            
            # Test original model
            with torch.no_grad():
                orig_pred = torch.argmax(original_model(x_noisy), dim=-1)
                if isinstance(y, torch.Tensor):
                    orig_correct = (orig_pred == y).float().item()
                else:
                    orig_correct = float(orig_pred.item() == y)
                original_robust_acc += orig_correct
            
            # Test resilient model
            with torch.no_grad():
                resilient_pred = torch.argmax(resilient_model(x_noisy), dim=-1)
                if isinstance(y, torch.Tensor):
                    resilient_correct = (resilient_pred == y).float().item()
                else:
                    resilient_correct = float(resilient_pred.item() == y)
                resilient_robust_acc += resilient_correct
        
        original_robust_acc /= len(test_samples)
        resilient_robust_acc /= len(test_samples)
        
        improvement = resilient_robust_acc - original_robust_acc
        return improvement
    
    def apply_to_model(self, model: Any) -> Any:
        """Apply resilience training to a model."""
        self.model = model
        return model
    
    def get_soft_predictions(self, x: torch.Tensor, use_temperature: bool = True) -> torch.Tensor:
        """Get soft predictions from the model."""
        
        with torch.no_grad():
            logits = self.model(x)
            
            if use_temperature and hasattr(self, 'temperature'):
                soft_preds = F.softmax(logits / self.temperature, dim=-1)
            else:
                soft_preds = F.softmax(logits, dim=-1)
            
            return soft_preds