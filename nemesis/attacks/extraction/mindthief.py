"""
MindThief Attack 🧠

"Stealing the very thoughts and memories of the machine"

Implementation of model extraction attacks that steal model parameters
or create functionally equivalent models through query-based learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Optional, Dict, List, Tuple, Callable
from ..base import AttackBase, AttackResult

class MindThief(AttackBase):
    """
    MindThief - Model Extraction Attack
    
    Like a thief who steals memories and thoughts, this attack
    extracts model knowledge by observing input-output relationships.
    """
    
    def __init__(self, model: Any):
        super().__init__(
            model=model,
            name="MindThief",
            description="Model extraction attack that steals knowledge through clever queries"
        )
        self.stolen_knowledge = []
        self.surrogate_model = None
    
    def forge_attack(self, x: Any, y: Any = None,
                    input_generator: Optional[Callable[[], Any]] = None,
                    surrogate_architecture: Optional[nn.Module] = None,
                    num_queries: int = 10000,
                    training_epochs: int = 50,
                    query_strategy: str = "random",
                    learning_rate: float = 0.001) -> AttackResult:
        """
        Forge a mind theft attack by extracting model knowledge.
        
        Args:
            x: Input sample or dataset
            y: Labels (optional)
            input_generator: Function that generates query inputs
            surrogate_architecture: Architecture for surrogate model
            num_queries: Number of queries to make to target model
            training_epochs: Epochs to train surrogate model
            query_strategy: Strategy for generating queries ("random", "adversarial", "hybrid")
            learning_rate: Learning rate for surrogate training
            
        Returns:
            AttackResult containing the stolen knowledge
        """
        
        # Handle case where input_generator is not provided
        if input_generator is None:
            if hasattr(x, 'shape'):
                # Create a simple generator based on input shape
                def simple_generator():
                    return torch.randn_like(x) if hasattr(torch, 'randn_like') else x
                input_generator = simple_generator
            else:
                # Default generator
                def default_generator():
                    return torch.randn(1, 3, 32, 32)
                input_generator = default_generator
        
        if self.framework == 'pytorch':
            return self._pytorch_mindthief(
                input_generator, surrogate_architecture, num_queries,
                training_epochs, query_strategy, learning_rate
            )
        else:
            raise NotImplementedError(f"MindThief not implemented for {self.framework}")
    
    def _pytorch_mindthief(self, input_generator: Callable[[], Any],
                          surrogate_architecture: Optional[nn.Module],
                          num_queries: int, training_epochs: int,
                          query_strategy: str, learning_rate: float) -> AttackResult:
        """PyTorch implementation of MindThief attack."""
        
        # Step 1: Generate queries and collect responses
        query_data = []
        
        for i in range(num_queries):
            if query_strategy == "random":
                query_input = input_generator()
            elif query_strategy == "adversarial":
                query_input = self._generate_adversarial_query(input_generator, i)
            elif query_strategy == "hybrid":
                if i % 2 == 0:
                    query_input = input_generator()
                else:
                    query_input = self._generate_adversarial_query(input_generator, i)
            else:
                query_input = input_generator()
            
            # Query the target model
            with torch.no_grad():
                response = self._predict(query_input)
            
            query_data.append((query_input, response))
            
            # Progress tracking
            if (i + 1) % 1000 == 0:
                print(f"Collected {i + 1}/{num_queries} query responses...")
        
        # Step 2: Create or use provided surrogate model
        if surrogate_architecture is None:
            surrogate_architecture = self._create_default_surrogate(query_data[0][0])
        
        self.surrogate_model = surrogate_architecture
        
        # Step 3: Train surrogate model on stolen data
        optimizer = optim.Adam(self.surrogate_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        training_losses = []
        
        for epoch in range(training_epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            np.random.shuffle(query_data)
            
            for query_input, target_response in query_data:
                optimizer.zero_grad()
                
                # Forward pass through surrogate
                surrogate_output = self.surrogate_model(query_input)
                
                # Convert target response to labels if needed
                if len(target_response.shape) > 1:
                    target_labels = torch.argmax(target_response, dim=-1)
                else:
                    target_labels = target_response.long()
                
                # Compute loss
                loss = criterion(surrogate_output, target_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(query_data)
            training_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{training_epochs}, Loss: {avg_loss:.4f}")
        
        # Step 4: Evaluate extraction success
        fidelity = self._compute_model_fidelity(query_data)
        
        # Create result
        success = fidelity > 0.8  # Consider successful if >80% fidelity
        
        return AttackResult(
            original_input=None,  # Not applicable for model extraction
            adversarial_input=self.surrogate_model,
            original_prediction=None,
            adversarial_prediction=None,
            perturbation=None,
            success=success,
            queries_used=num_queries,
            perturbation_norm=0.0,
            confidence_drop=0.0,
            attack_name=self.name,
            metadata={
                'num_queries': num_queries,
                'training_epochs': training_epochs,
                'query_strategy': query_strategy,
                'learning_rate': learning_rate,
                'model_fidelity': fidelity,
                'training_losses': training_losses,
                'surrogate_parameters': sum(p.numel() for p in self.surrogate_model.parameters()),
            }
        )
    
    def _generate_adversarial_query(self, input_generator: Callable[[], Any], 
                                   iteration: int) -> torch.Tensor:
        """Generate adversarial queries to maximize information extraction."""
        
        # Start with random input
        base_input = input_generator()
        
        # Add some entropy to explore different regions
        if iteration < 1000:
            # Early iterations: pure random
            return base_input
        elif iteration < 5000:
            # Middle iterations: add noise to previous successful queries
            if self.stolen_knowledge:
                prev_input = self.stolen_knowledge[-1][0]
                noise_scale = 0.1
                noise = torch.randn_like(prev_input) * noise_scale
                return torch.clamp(prev_input + noise, 0, 1)
            else:
                return base_input
        else:
            # Later iterations: focus on decision boundaries
            return self._boundary_query(base_input)
    
    def _boundary_query(self, base_input: torch.Tensor) -> torch.Tensor:
        """Generate queries near decision boundaries."""
        
        # Simple approach: add small perturbations and check if prediction changes
        perturbations = torch.randn_like(base_input) * 0.05
        boundary_input = torch.clamp(base_input + perturbations, 0, 1)
        
        return boundary_input
    
    def _create_default_surrogate(self, sample_input: torch.Tensor) -> nn.Module:
        """Create a default surrogate model architecture."""
        
        input_size = sample_input.numel()
        
        # Simple feedforward network
        surrogate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Assume 10 classes, adjust as needed
        )
        
        return surrogate
    
    def _compute_model_fidelity(self, test_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Compute fidelity between original and surrogate model."""
        
        if self.surrogate_model is None:
            return 0.0
        
        correct_predictions = 0
        total_predictions = len(test_data)
        
        with torch.no_grad():
            for query_input, original_output in test_data:
                # Get original prediction
                original_pred = torch.argmax(original_output, dim=-1)
                
                # Get surrogate prediction
                surrogate_output = self.surrogate_model(query_input)
                surrogate_pred = torch.argmax(surrogate_output, dim=-1)
                
                if original_pred == surrogate_pred:
                    correct_predictions += 1
        
        fidelity = correct_predictions / total_predictions
        return fidelity
    
    def extract_gradients(self, input_sample: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Attempt to extract gradient information (for white-box scenarios).
        
        Args:
            input_sample: Sample input for gradient computation
            
        Returns:
            Dictionary containing extracted gradient information
        """
        
        gradients = {}
        
        # Enable gradient computation
        input_sample.requires_grad_(True)
        
        # Forward pass
        output = self._predict(input_sample)
        
        # Compute gradients for each output class
        for class_idx in range(output.shape[-1]):
            # Zero gradients
            if input_sample.grad is not None:
                input_sample.grad.zero_()
            
            # Backward pass for this class
            class_output = output[0, class_idx]
            class_output.backward(retain_graph=True)
            
            # Store gradient
            gradients[f'class_{class_idx}'] = input_sample.grad.clone()
        
        return gradients
    
    def membership_inference(self, candidate_inputs: List[torch.Tensor],
                           confidence_threshold: float = 0.9) -> List[bool]:
        """
        Perform membership inference to determine if inputs were in training set.
        
        Args:
            candidate_inputs: Inputs to test for membership
            confidence_threshold: Threshold for membership decision
            
        Returns:
            List of boolean membership decisions
        """
        
        membership_decisions = []
        
        with torch.no_grad():
            for input_tensor in candidate_inputs:
                # Get model confidence
                output = self._predict(input_tensor)
                max_confidence = torch.max(torch.softmax(output, dim=-1)).item()
                
                # High confidence might indicate training set membership
                is_member = max_confidence > confidence_threshold
                membership_decisions.append(is_member)
        
        return membership_decisions
    
    def save_stolen_model(self, filepath: str):
        """Save the extracted surrogate model."""
        if self.surrogate_model is not None:
            torch.save(self.surrogate_model.state_dict(), filepath)
            print(f"Stolen model saved to {filepath}")
        else:
            print("No surrogate model to save. Run extraction attack first.")
    
    def load_stolen_model(self, filepath: str, model_architecture: nn.Module):
        """Load a previously extracted model."""
        model_architecture.load_state_dict(torch.load(filepath))
        self.surrogate_model = model_architecture
        print(f"Stolen model loaded from {filepath}")
    
    def compare_models(self, test_inputs: List[torch.Tensor]) -> Dict[str, float]:
        """
        Compare performance between original and stolen model.
        
        Args:
            test_inputs: Test inputs for comparison
            
        Returns:
            Comparison metrics
        """
        
        if self.surrogate_model is None:
            return {'error': 'No surrogate model available'}
        
        agreement_count = 0
        total_count = len(test_inputs)
        
        confidence_differences = []
        
        with torch.no_grad():
            for test_input in test_inputs:
                # Original model prediction
                original_output = self._predict(test_input)
                original_pred = torch.argmax(original_output, dim=-1)
                original_conf = torch.max(torch.softmax(original_output, dim=-1))
                
                # Surrogate model prediction
                surrogate_output = self.surrogate_model(test_input)
                surrogate_pred = torch.argmax(surrogate_output, dim=-1)
                surrogate_conf = torch.max(torch.softmax(surrogate_output, dim=-1))
                
                # Check agreement
                if original_pred == surrogate_pred:
                    agreement_count += 1
                
                # Confidence difference
                conf_diff = abs(original_conf.item() - surrogate_conf.item())
                confidence_differences.append(conf_diff)
        
        fidelity = agreement_count / total_count
        avg_conf_diff = np.mean(confidence_differences)
        
        return {
            'fidelity': fidelity,
            'agreement_count': agreement_count,
            'total_samples': total_count,
            'average_confidence_difference': avg_conf_diff,
            'extraction_quality': 'high' if fidelity > 0.9 else 'medium' if fidelity > 0.7 else 'low'
        }