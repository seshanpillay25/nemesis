"""
Corruption Attack 🦠

"Like a plague that spreads through the land, corrupting all it touches"

Implementation of label flipping and data corruption attacks that
systematically degrade model performance through training data manipulation.
"""

import numpy as np
import torch
from typing import Any, Optional, Dict, List, Tuple, Union
from ..base import AttackBase, AttackResult

class Corruption(AttackBase):
    """
    Corruption - Label Flipping and Data Corruption Attack
    
    Like a plague that spreads corruption wherever it goes,
    this attack systematically degrades model performance.
    """
    
    def __init__(self, model: Any):
        super().__init__(
            model=model,
            name="Corruption",
            description="Systematic corruption of training data to degrade model performance"
        )
        self.corruption_history = []
    
    def forge_attack(self, x: Any, y: Any = None,
                    dataset: Optional[List[Tuple[Any, Any]]] = None,
                    corruption_type: str = "label_flip",
                    corruption_rate: float = 0.1,
                    target_classes: Optional[List[int]] = None,
                    flip_pairs: Optional[List[Tuple[int, int]]] = None,
                    noise_std: float = 0.1) -> AttackResult:
        """
        Forge a corruption attack on a dataset.
        
        Args:
            x: Dataset or input data
            y: Labels (optional)
            dataset: List of (input, label) tuples (alternative to x)
            corruption_type: Type of corruption ("label_flip", "noise_injection", "mixed")
            corruption_rate: Fraction of dataset to corrupt
            target_classes: Specific classes to target for corruption
            flip_pairs: Specific label flip pairs [(from_class, to_class), ...]
            noise_std: Standard deviation for noise injection
            
        Returns:
            AttackResult containing the corrupted dataset
        """
        # Handle different input formats
        if dataset is None:
            # If x is a list/dataset, use it directly
            if isinstance(x, list):
                dataset = x
            else:
                # Create dataset from x, y
                dataset = [(x, y)] if y is not None else [(x, 0)]
        
        if corruption_type == "label_flip":
            return self._label_flip_corruption(
                dataset, corruption_rate, target_classes, flip_pairs
            )
        elif corruption_type == "noise_injection":
            return self._noise_injection_corruption(
                dataset, corruption_rate, noise_std, target_classes
            )
        elif corruption_type == "mixed":
            return self._mixed_corruption(
                dataset, corruption_rate, target_classes, flip_pairs, noise_std
            )
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
    
    def _label_flip_corruption(self, dataset: List[Tuple[Any, Any]],
                              corruption_rate: float,
                              target_classes: Optional[List[int]],
                              flip_pairs: Optional[List[Tuple[int, int]]]) -> AttackResult:
        """Implement label flipping corruption."""
        
        original_dataset = dataset.copy()
        corrupted_dataset = []
        
        # Determine classes in dataset
        all_labels = [label for _, label in dataset]
        unique_classes = list(set(all_labels))
        
        # Set up flip mapping
        if flip_pairs:
            flip_mapping = dict(flip_pairs)
        else:
            # Default: create random flip mapping
            flip_mapping = {}
            for cls in unique_classes:
                if target_classes is None or cls in target_classes:
                    # Flip to a random different class
                    other_classes = [c for c in unique_classes if c != cls]
                    if other_classes:
                        flip_mapping[cls] = np.random.choice(other_classes)
        
        # Select samples to corrupt
        num_to_corrupt = int(len(dataset) * corruption_rate)
        
        if target_classes:
            # Only corrupt samples from target classes
            target_indices = [i for i, (_, label) in enumerate(dataset) 
                             if label in target_classes]
            if len(target_indices) < num_to_corrupt:
                corrupt_indices = target_indices
            else:
                corrupt_indices = np.random.choice(
                    target_indices, num_to_corrupt, replace=False
                )
        else:
            # Corrupt random samples
            corrupt_indices = np.random.choice(
                len(dataset), num_to_corrupt, replace=False
            )
        
        # Apply corruption
        corrupted_count = 0
        for i, (x, y) in enumerate(dataset):
            if i in corrupt_indices and y in flip_mapping:
                # Flip label
                new_label = flip_mapping[y]
                corrupted_dataset.append((x, new_label))
                corrupted_count += 1
            else:
                # Keep original
                corrupted_dataset.append((x, y))
        
        # Create result
        success = corrupted_count > 0
        
        return AttackResult(
            original_input=original_dataset,
            adversarial_input=corrupted_dataset,
            original_prediction=None,  # Not applicable for dataset corruption
            adversarial_prediction=None,
            perturbation=None,
            success=success,
            queries_used=0,  # No model queries needed
            perturbation_norm=0.0,  # Label changes don't have norm
            confidence_drop=0.0,
            attack_name=self.name,
            metadata={
                'corruption_type': 'label_flip',
                'corruption_rate': corruption_rate,
                'requested_corruptions': num_to_corrupt,
                'actual_corruptions': corrupted_count,
                'flip_mapping': flip_mapping,
                'target_classes': target_classes,
                'dataset_size': len(dataset),
            }
        )
    
    def _noise_injection_corruption(self, dataset: List[Tuple[Any, Any]],
                                   corruption_rate: float, noise_std: float,
                                   target_classes: Optional[List[int]]) -> AttackResult:
        """Implement noise injection corruption."""
        
        original_dataset = dataset.copy()
        corrupted_dataset = []
        
        # Select samples to corrupt
        num_to_corrupt = int(len(dataset) * corruption_rate)
        
        if target_classes:
            target_indices = [i for i, (_, label) in enumerate(dataset) 
                             if label in target_classes]
            if len(target_indices) < num_to_corrupt:
                corrupt_indices = target_indices
            else:
                corrupt_indices = np.random.choice(
                    target_indices, num_to_corrupt, replace=False
                )
        else:
            corrupt_indices = np.random.choice(
                len(dataset), num_to_corrupt, replace=False
            )
        
        # Apply corruption
        corrupted_count = 0
        total_noise_norm = 0.0
        
        for i, (x, y) in enumerate(dataset):
            if i in corrupt_indices:
                # Add noise to input
                if isinstance(x, torch.Tensor):
                    noise = torch.randn_like(x) * noise_std
                    corrupted_x = x + noise
                    noise_norm = torch.norm(noise).item()
                elif isinstance(x, np.ndarray):
                    noise = np.random.randn(*x.shape) * noise_std
                    corrupted_x = x + noise
                    noise_norm = np.linalg.norm(noise)
                else:
                    # Skip if input type not supported
                    corrupted_dataset.append((x, y))
                    continue
                
                corrupted_dataset.append((corrupted_x, y))
                corrupted_count += 1
                total_noise_norm += noise_norm
            else:
                corrupted_dataset.append((x, y))
        
        # Calculate average noise norm
        avg_noise_norm = total_noise_norm / max(corrupted_count, 1)
        
        success = corrupted_count > 0
        
        return AttackResult(
            original_input=original_dataset,
            adversarial_input=corrupted_dataset,
            original_prediction=None,
            adversarial_prediction=None,
            perturbation=None,
            success=success,
            queries_used=0,
            perturbation_norm=avg_noise_norm,
            confidence_drop=0.0,
            attack_name=self.name,
            metadata={
                'corruption_type': 'noise_injection',
                'corruption_rate': corruption_rate,
                'noise_std': noise_std,
                'requested_corruptions': num_to_corrupt,
                'actual_corruptions': corrupted_count,
                'average_noise_norm': avg_noise_norm,
                'target_classes': target_classes,
                'dataset_size': len(dataset),
            }
        )
    
    def _mixed_corruption(self, dataset: List[Tuple[Any, Any]],
                         corruption_rate: float,
                         target_classes: Optional[List[int]],
                         flip_pairs: Optional[List[Tuple[int, int]]],
                         noise_std: float) -> AttackResult:
        """Implement mixed corruption (both label flipping and noise injection)."""
        
        # Split corruption budget between label flipping and noise injection
        label_corruption_rate = corruption_rate * 0.6  # 60% for label flipping
        noise_corruption_rate = corruption_rate * 0.4   # 40% for noise injection
        
        # Apply label flipping first
        label_result = self._label_flip_corruption(
            dataset, label_corruption_rate, target_classes, flip_pairs
        )
        
        # Then apply noise injection to the already corrupted dataset
        noise_result = self._noise_injection_corruption(
            label_result.adversarial_input, noise_corruption_rate, 
            noise_std, target_classes
        )
        
        # Combine metadata
        combined_metadata = {
            'corruption_type': 'mixed',
            'total_corruption_rate': corruption_rate,
            'label_corruption': label_result.metadata,
            'noise_corruption': noise_result.metadata,
            'dataset_size': len(dataset),
        }
        
        return AttackResult(
            original_input=dataset,
            adversarial_input=noise_result.adversarial_input,
            original_prediction=None,
            adversarial_prediction=None,
            perturbation=None,
            success=label_result.success or noise_result.success,
            queries_used=0,
            perturbation_norm=noise_result.perturbation_norm,
            confidence_drop=0.0,
            attack_name=self.name,
            metadata=combined_metadata
        )
    
    def analyze_corruption_impact(self, clean_model: Any, corrupted_model: Any,
                                 test_dataset: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        """
        Analyze the impact of corruption on model performance.
        
        Args:
            clean_model: Model trained on clean data
            corrupted_model: Model trained on corrupted data
            test_dataset: Clean test dataset
            
        Returns:
            Analysis of corruption impact
        """
        clean_accuracy = self._evaluate_model(clean_model, test_dataset)
        corrupted_accuracy = self._evaluate_model(corrupted_model, test_dataset)
        
        accuracy_drop = clean_accuracy - corrupted_accuracy
        relative_drop = accuracy_drop / clean_accuracy if clean_accuracy > 0 else 0
        
        return {
            'clean_accuracy': clean_accuracy,
            'corrupted_accuracy': corrupted_accuracy,
            'accuracy_drop': accuracy_drop,
            'relative_accuracy_drop': relative_drop,
            'corruption_effectiveness': relative_drop,
            'test_samples': len(test_dataset)
        }
    
    def _evaluate_model(self, model: Any, test_dataset: List[Tuple[Any, Any]]) -> float:
        """Evaluate model accuracy on test dataset."""
        correct = 0
        total = len(test_dataset)
        
        for x, y in test_dataset:
            pred = self._predict_with_model(model, x)
            
            if isinstance(pred, torch.Tensor):
                pred_class = torch.argmax(pred, dim=-1).item()
            else:
                pred_class = np.argmax(pred)
            
            if pred_class == y:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _predict_with_model(self, model: Any, x: Any) -> Any:
        """Make prediction with specific model."""
        if hasattr(model, 'predict'):
            return model.predict(x if isinstance(x, np.ndarray) else x.numpy())
        else:
            # Assume PyTorch or TensorFlow model
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            
            if hasattr(model, 'forward'):  # PyTorch
                with torch.no_grad():
                    return model(x)
            else:  # TensorFlow
                return model(x)
    
    def create_targeted_corruption(self, dataset: List[Tuple[Any, Any]],
                                  source_class: int, target_class: int,
                                  corruption_rate: float = 1.0) -> List[Tuple[Any, Any]]:
        """
        Create targeted corruption that flips all instances of source_class to target_class.
        
        Args:
            dataset: Original dataset
            source_class: Class to corrupt
            target_class: Class to flip to
            corruption_rate: Fraction of source class samples to corrupt
            
        Returns:
            Corrupted dataset
        """
        result = self.forge_attack(
            dataset=dataset,
            corruption_type="label_flip",
            corruption_rate=corruption_rate,
            target_classes=[source_class],
            flip_pairs=[(source_class, target_class)]
        )
        
        return result.adversarial_input
    
    def get_corruption_statistics(self) -> Dict[str, Any]:
        """Get statistics about corruption attacks performed."""
        if not self.corruption_history:
            return {'message': 'No corruptions performed yet'}
        
        stats = {
            'total_corruptions': len(self.corruption_history),
            'corruption_types': {},
            'average_corruption_rate': 0.0,
            'total_samples_corrupted': 0
        }
        
        for corruption in self.corruption_history:
            corruption_type = corruption.get('corruption_type', 'unknown')
            if corruption_type not in stats['corruption_types']:
                stats['corruption_types'][corruption_type] = 0
            stats['corruption_types'][corruption_type] += 1
            
            stats['average_corruption_rate'] += corruption.get('corruption_rate', 0)
            stats['total_samples_corrupted'] += corruption.get('actual_corruptions', 0)
        
        stats['average_corruption_rate'] /= len(self.corruption_history)
        
        return stats