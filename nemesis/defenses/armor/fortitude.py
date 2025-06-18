"""
Fortitude Armor ⚔️

"Strength forged through endless battles"

Implementation of adversarial training defenses that strengthen
models by exposing them to adversarial examples during training.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Optional, Dict, List, Tuple
from ..base import DefenseBase, DefenseResult

logger = logging.getLogger(__name__)

class Fortitude(DefenseBase):
    """
    Fortitude - Adversarial Training Defense
    
    Like a warrior who grows stronger through battle,
    Fortitude hardens models against attacks through adversarial training.
    """
    
    def __init__(self, model: Any):
        super().__init__(
            name="Fortitude",
            description="Adversarial training that forges strength through battle",
            defense_type="training"
        )
        self.model = model
        self.training_history = []
    
    def forge_defense(self, training_data: List[Tuple[Any, Any]],
                     attack_method: Any = None, epsilon: float = 0.1,
                     alpha: float = 0.01, num_steps: int = 7,
                     training_epochs: int = 10, **kwargs) -> DefenseResult:
        """
        Forge Fortitude defense through adversarial training.
        
        Args:
            training_data: List of (input, label) training pairs
            attack_method: Attack method to use for generating adversarial examples
            epsilon: Maximum perturbation magnitude
            alpha: Step size for adversarial generation
            num_steps: Number of attack steps
            training_epochs: Number of training epochs
            **kwargs: Additional training parameters
            
        Returns:
            DefenseResult containing the hardened model
        """
        
        original_model = self._clone_model(self.model)
        
        # Perform adversarial training
        hardened_model = self._adversarial_training(
            training_data, attack_method, epsilon, alpha, 
            num_steps, training_epochs, **kwargs
        )
        
        # Evaluate improvement
        improvement_score = self._evaluate_robustness_improvement(
            original_model, hardened_model, training_data
        )
        
        defense_applied = improvement_score > 0.05  # 5% improvement threshold
        
        return DefenseResult(
            original_input=original_model,
            defended_input=hardened_model,
            original_prediction=None,
            defended_prediction=None,
            defense_applied=defense_applied,
            confidence_change=0.0,
            defense_strength=improvement_score,
            defense_name=self.name,
            metadata={
                'training_method': 'adversarial_training',
                'epsilon': epsilon,
                'alpha': alpha,
                'num_steps': num_steps,
                'training_epochs': training_epochs,
                'robustness_improvement': improvement_score,
                'training_samples': len(training_data),
            }
        )
    
    def _adversarial_training(self, training_data: List[Tuple[Any, Any]],
                             attack_method: Any, epsilon: float, alpha: float,
                             num_steps: int, training_epochs: int, **kwargs) -> Any:
        """Perform adversarial training on the model."""
        
        # Handle empty dataset
        if len(training_data) == 0:
            logger.warning("Empty training dataset provided. Returning original model.")
            return self.model
        
        # Set up optimizer
        learning_rate = kwargs.get('learning_rate', 0.001)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(training_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            
            # Shuffle training data
            import random
            random.shuffle(training_data)
            
            for x, y in training_data:
                optimizer.zero_grad()
                
                # Generate adversarial examples
                if attack_method is not None:
                    try:
                        adv_result = attack_method.unleash(x, y, epsilon=epsilon)
                        x_adv = adv_result.adversarial_input
                    except:
                        # Fallback to PGD if attack method fails
                        x_adv = self._generate_pgd_adversarial(x, y, epsilon, alpha, num_steps)
                else:
                    # Use PGD as default
                    x_adv = self._generate_pgd_adversarial(x, y, epsilon, alpha, num_steps)
                
                # Mixed training: half clean, half adversarial
                if num_batches % 2 == 0:
                    training_input = x
                else:
                    training_input = x_adv
                
                # Forward pass
                output = self.model(training_input)
                
                # Compute loss
                if isinstance(y, torch.Tensor):
                    loss = criterion(output, y)
                else:
                    y_tensor = torch.tensor([y], dtype=torch.long)
                    loss = criterion(output, y_tensor)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                pred = torch.argmax(output, dim=-1)
                if isinstance(y, torch.Tensor):
                    correct = (pred == y).float().mean()
                else:
                    correct = (pred.item() == y)
                epoch_acc += float(correct)
                num_batches += 1
            
            # Log epoch results
            avg_loss = epoch_loss / max(num_batches, 1)  # Avoid division by zero
            avg_acc = epoch_acc / max(num_batches, 1)
            
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': avg_acc
            })
            
            print(f"Epoch {epoch+1}/{training_epochs}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        
        return self.model
    
    def _generate_pgd_adversarial(self, x: torch.Tensor, y: Any,
                                 epsilon: float, alpha: float, num_steps: int) -> torch.Tensor:
        """Generate adversarial examples using PGD."""
        
        # Initialize adversarial input
        x_adv = x.clone()
        
        # Random initialization
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0, 1)
        
        # PGD iterations
        for _ in range(num_steps):
            x_adv.requires_grad_(True)
            
            # Forward pass
            output = self.model(x_adv)
            
            # Compute loss
            if isinstance(y, torch.Tensor):
                loss = nn.functional.cross_entropy(output, y)
            else:
                y_tensor = torch.tensor([y], dtype=torch.long)
                loss = nn.functional.cross_entropy(output, y_tensor)
            
            # Backward pass
            loss.backward()
            
            # Update adversarial input
            grad_sign = torch.sign(x_adv.grad.data)
            x_adv = x_adv.detach() + alpha * grad_sign
            
            # Project back to epsilon ball
            delta = x_adv - x
            delta = torch.clamp(delta, -epsilon, epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()
    
    def _clone_model(self, model: Any) -> Any:
        """Create a copy of the model for comparison."""
        try:
            import copy
            return copy.deepcopy(model)
        except:
            # If deep copy fails, return reference
            return model
    
    def _evaluate_robustness_improvement(self, original_model: Any, 
                                       hardened_model: Any,
                                       test_data: List[Tuple[Any, Any]]) -> float:
        """Evaluate how much the robustness improved."""
        
        # Simple evaluation: compare accuracy on adversarial examples
        if len(test_data) == 0:
            return 0.0
        
        # Generate some adversarial test examples
        test_sample_size = min(10, len(test_data))
        test_samples = test_data[:test_sample_size]
        
        original_robust_acc = 0.0
        hardened_robust_acc = 0.0
        
        for x, y in test_samples:
            # Generate adversarial example
            x_adv = self._generate_pgd_adversarial(x, y, 0.1, 0.01, 7)
            
            # Test original model
            with torch.no_grad():
                orig_pred = torch.argmax(original_model(x_adv), dim=-1)
                if isinstance(y, torch.Tensor):
                    orig_correct = (orig_pred == y).float().item()
                else:
                    orig_correct = float(orig_pred.item() == y)
                original_robust_acc += orig_correct
            
            # Test hardened model
            with torch.no_grad():
                hard_pred = torch.argmax(hardened_model(x_adv), dim=-1)
                if isinstance(y, torch.Tensor):
                    hard_correct = (hard_pred == y).float().item()
                else:
                    hard_correct = float(hard_pred.item() == y)
                hardened_robust_acc += hard_correct
        
        original_robust_acc /= test_sample_size
        hardened_robust_acc /= test_sample_size
        
        improvement = hardened_robust_acc - original_robust_acc
        return improvement
    
    def apply_to_model(self, model: Any) -> Any:
        """Apply fortitude training to a model."""
        self.model = model
        # Return the model (training happens during forge_defense)
        return model
    
    def trades_training(self, training_data: List[Tuple[Any, Any]],
                       beta: float = 6.0, training_epochs: int = 10) -> DefenseResult:
        """
        Implement TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization).
        
        Args:
            training_data: Training dataset
            beta: Trade-off parameter between natural and robust accuracy
            training_epochs: Number of training epochs
            
        Returns:
            DefenseResult with TRADES-trained model
        """
        
        original_model = self._clone_model(self.model)
        
        # TRADES training
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(training_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for x, y in training_data:
                optimizer.zero_grad()
                
                # Natural loss
                natural_output = self.model(x)
                if isinstance(y, torch.Tensor):
                    natural_loss = nn.functional.cross_entropy(natural_output, y)
                else:
                    y_tensor = torch.tensor([y], dtype=torch.long)
                    natural_loss = nn.functional.cross_entropy(natural_output, y_tensor)
                
                # Generate adversarial examples
                x_adv = self._generate_pgd_adversarial(x, y, 0.1, 0.01, 7)
                
                # Adversarial output
                adv_output = self.model(x_adv)
                
                # KL divergence loss (TRADES loss)
                kl_loss = nn.functional.kl_div(
                    nn.functional.log_softmax(adv_output, dim=1),
                    nn.functional.softmax(natural_output, dim=1),
                    reduction='batchmean'
                )
                
                # Total loss
                total_loss = natural_loss + beta * kl_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"TRADES Epoch {epoch+1}/{training_epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluate improvement
        improvement = self._evaluate_robustness_improvement(
            original_model, self.model, training_data
        )
        
        return DefenseResult(
            original_input=original_model,
            defended_input=self.model,
            original_prediction=None,
            defended_prediction=None,
            defense_applied=improvement > 0.05,
            confidence_change=0.0,
            defense_strength=improvement,
            defense_name=f"{self.name}-TRADES",
            metadata={
                'training_method': 'TRADES',
                'beta': beta,
                'training_epochs': training_epochs,
                'robustness_improvement': improvement,
            }
        )
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get the training history."""
        return self.training_history