"""
Oracle Attack 🔮

"The oracle speaks truth through careful questioning"

Implementation of query-based extraction attacks that gather information
about models through strategic querying patterns and analysis.
"""

import numpy as np
import torch
from typing import Any, Optional, Dict, List, Tuple, Callable
import itertools
from collections import defaultdict
from ..base import AttackBase, AttackResult

class Oracle(AttackBase):
    """
    Oracle - Query-based Information Extraction
    
    Like the ancient oracles who revealed hidden knowledge through
    cryptic responses, this attack extracts secrets through careful questioning.
    """
    
    def __init__(self, model: Any):
        super().__init__(
            model=model,
            name="Oracle",
            description="Query-based extraction that reveals model secrets through strategic questioning"
        )
        self.query_log = []
        self.extracted_knowledge = {}
    
    def forge_attack(self, x: Any, y: Any = None,
                    attack_type: str = "decision_boundary",
                    num_queries: int = 1000,
                    input_generator: Optional[Callable[[], Any]] = None,
                    target_info: str = "architecture") -> AttackResult:
        """
        Forge an oracle attack to extract specific information.
        
        Args:
            x: Input sample or dataset
            y: Labels (optional) 
            attack_type: Type of oracle attack ("decision_boundary", "confidence", "gradient")
            num_queries: Number of queries to perform
            input_generator: Function to generate query inputs
            target_info: Type of information to extract ("architecture", "training_data", "parameters")
            
        Returns:
            AttackResult containing extracted information
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
        
        if attack_type == "decision_boundary":
            return self._decision_boundary_oracle(num_queries, input_generator)
        elif attack_type == "confidence":
            return self._confidence_oracle(num_queries, input_generator)
        elif attack_type == "gradient":
            return self._gradient_oracle(num_queries, input_generator)
        elif attack_type == "comprehensive":
            return self._comprehensive_oracle(num_queries, input_generator, target_info)
        else:
            raise ValueError(f"Unknown oracle attack type: {attack_type}")
    
    def _decision_boundary_oracle(self, num_queries: int,
                                 input_generator: Optional[Callable[[], Any]]) -> AttackResult:
        """Extract information about decision boundaries."""
        
        boundary_info = {
            'boundary_points': [],
            'class_regions': defaultdict(list),
            'uncertainty_regions': []
        }
        
        for i in range(num_queries):
            # Generate query point
            if input_generator:
                query_point = input_generator()
            else:
                query_point = self._generate_random_input()
            
            # Get model response
            response = self._predict(query_point)
            predicted_class = torch.argmax(response, dim=-1).item()
            confidence = torch.max(torch.softmax(response, dim=-1)).item()
            
            # Log query
            self.query_log.append({
                'input': query_point.clone(),
                'response': response.clone(),
                'predicted_class': predicted_class,
                'confidence': confidence
            })
            
            # Analyze response
            boundary_info['class_regions'][predicted_class].append(query_point)
            
            # Identify potential boundary points (low confidence)
            if confidence < 0.6:
                boundary_info['uncertainty_regions'].append({
                    'point': query_point,
                    'confidence': confidence,
                    'predicted_class': predicted_class
                })
            
            # Try to find exact boundary points
            if i > 100 and len(boundary_info['uncertainty_regions']) > 0:
                boundary_point = self._refine_boundary_point(
                    boundary_info['uncertainty_regions'][-1]['point']
                )
                if boundary_point is not None:
                    boundary_info['boundary_points'].append(boundary_point)
        
        # Analyze extracted information
        num_classes_found = len(boundary_info['class_regions'])
        num_boundaries_found = len(boundary_info['boundary_points'])
        avg_confidence = np.mean([log['confidence'] for log in self.query_log])
        
        success = num_classes_found > 1 and num_boundaries_found > 0
        
        self.extracted_knowledge['decision_boundaries'] = boundary_info
        
        return AttackResult(
            original_input=None,
            adversarial_input=boundary_info,
            original_prediction=None,
            adversarial_prediction=None,
            perturbation=None,
            success=success,
            queries_used=num_queries,
            perturbation_norm=0.0,
            confidence_drop=0.0,
            attack_name=self.name,
            metadata={
                'attack_type': 'decision_boundary',
                'num_classes_discovered': num_classes_found,
                'num_boundary_points': num_boundaries_found,
                'average_confidence': avg_confidence,
                'uncertainty_regions': len(boundary_info['uncertainty_regions']),
            }
        )
    
    def _confidence_oracle(self, num_queries: int,
                          input_generator: Optional[Callable[[], Any]]) -> AttackResult:
        """Extract information about model confidence patterns."""
        
        confidence_info = {
            'high_confidence_regions': [],
            'low_confidence_regions': [],
            'confidence_distribution': [],
            'class_confidence_patterns': defaultdict(list)
        }
        
        for i in range(num_queries):
            # Generate query
            if input_generator:
                query_point = input_generator()
            else:
                query_point = self._generate_random_input()
            
            # Get response
            response = self._predict(query_point)
            predicted_class = torch.argmax(response, dim=-1).item()
            confidence = torch.max(torch.softmax(response, dim=-1)).item()
            
            # Store confidence information
            confidence_info['confidence_distribution'].append(confidence)
            confidence_info['class_confidence_patterns'][predicted_class].append(confidence)
            
            # Categorize by confidence level
            if confidence > 0.9:
                confidence_info['high_confidence_regions'].append({
                    'point': query_point,
                    'confidence': confidence,
                    'class': predicted_class
                })
            elif confidence < 0.6:
                confidence_info['low_confidence_regions'].append({
                    'point': query_point,
                    'confidence': confidence,
                    'class': predicted_class
                })
        
        # Analyze patterns
        avg_confidence = np.mean(confidence_info['confidence_distribution'])
        confidence_std = np.std(confidence_info['confidence_distribution'])
        
        # Per-class confidence analysis
        class_confidence_stats = {}
        for class_id, confidences in confidence_info['class_confidence_patterns'].items():
            class_confidence_stats[class_id] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'count': len(confidences)
            }
        
        success = len(confidence_info['high_confidence_regions']) > 0
        
        self.extracted_knowledge['confidence_patterns'] = confidence_info
        
        return AttackResult(
            original_input=None,
            adversarial_input=confidence_info,
            original_prediction=None,
            adversarial_prediction=None,
            perturbation=None,
            success=success,
            queries_used=num_queries,
            perturbation_norm=0.0,
            confidence_drop=0.0,
            attack_name=self.name,
            metadata={
                'attack_type': 'confidence',
                'average_confidence': avg_confidence,
                'confidence_std': confidence_std,
                'high_confidence_regions': len(confidence_info['high_confidence_regions']),
                'low_confidence_regions': len(confidence_info['low_confidence_regions']),
                'class_confidence_stats': class_confidence_stats,
            }
        )
    
    def _gradient_oracle(self, num_queries: int,
                        input_generator: Optional[Callable[[], Any]]) -> AttackResult:
        """Extract gradient information (if accessible)."""
        
        gradient_info = {
            'gradient_norms': [],
            'gradient_directions': [],
            'lipschitz_estimates': []
        }
        
        for i in range(num_queries):
            try:
                # Generate query point
                if input_generator:
                    query_point = input_generator()
                else:
                    query_point = self._generate_random_input()
                
                query_point.requires_grad_(True)
                
                # Forward pass
                response = self._predict(query_point)
                
                # Compute gradient for dominant class
                dominant_class = torch.argmax(response, dim=-1)
                loss = response[0, dominant_class]
                loss.backward()
                
                # Extract gradient information
                gradient = query_point.grad
                gradient_norm = torch.norm(gradient).item()
                
                gradient_info['gradient_norms'].append(gradient_norm)
                gradient_info['gradient_directions'].append(gradient.clone())
                
                # Estimate local Lipschitz constant
                if i > 0:
                    prev_point = self.query_log[-1]['input']
                    prev_response = self.query_log[-1]['response']
                    
                    input_diff = torch.norm(query_point - prev_point).item()
                    output_diff = torch.norm(response - prev_response).item()
                    
                    if input_diff > 0:
                        lipschitz_est = output_diff / input_diff
                        gradient_info['lipschitz_estimates'].append(lipschitz_est)
                
                # Log query
                self.query_log.append({
                    'input': query_point.detach().clone(),
                    'response': response.detach().clone(),
                    'gradient': gradient.clone(),
                    'gradient_norm': gradient_norm
                })
                
            except Exception as e:
                # Skip if gradient computation fails
                continue
        
        # Analyze gradient patterns
        if gradient_info['gradient_norms']:
            avg_grad_norm = np.mean(gradient_info['gradient_norms'])
            grad_norm_std = np.std(gradient_info['gradient_norms'])
        else:
            avg_grad_norm = 0.0
            grad_norm_std = 0.0
        
        if gradient_info['lipschitz_estimates']:
            avg_lipschitz = np.mean(gradient_info['lipschitz_estimates'])
        else:
            avg_lipschitz = 0.0
        
        success = len(gradient_info['gradient_norms']) > 0
        
        self.extracted_knowledge['gradients'] = gradient_info
        
        return AttackResult(
            original_input=None,
            adversarial_input=gradient_info,
            original_prediction=None,
            adversarial_prediction=None,
            perturbation=None,
            success=success,
            queries_used=num_queries,
            perturbation_norm=0.0,
            confidence_drop=0.0,
            attack_name=self.name,
            metadata={
                'attack_type': 'gradient',
                'average_gradient_norm': avg_grad_norm,
                'gradient_norm_std': grad_norm_std,
                'average_lipschitz': avg_lipschitz,
                'successful_gradient_queries': len(gradient_info['gradient_norms']),
            }
        )
    
    def _comprehensive_oracle(self, num_queries: int,
                             input_generator: Optional[Callable[[], Any]],
                             target_info: str) -> AttackResult:
        """Perform comprehensive information extraction."""
        
        # Divide queries among different attack types
        boundary_queries = num_queries // 3
        confidence_queries = num_queries // 3
        gradient_queries = num_queries - boundary_queries - confidence_queries
        
        # Perform sub-attacks
        boundary_result = self._decision_boundary_oracle(boundary_queries, input_generator)
        confidence_result = self._confidence_oracle(confidence_queries, input_generator)
        gradient_result = self._gradient_oracle(gradient_queries, input_generator)
        
        # Combine results
        comprehensive_info = {
            'boundaries': boundary_result.adversarial_input,
            'confidence': confidence_result.adversarial_input,
            'gradients': gradient_result.adversarial_input,
            'model_analysis': self._analyze_model_properties()
        }
        
        overall_success = (boundary_result.success or 
                          confidence_result.success or 
                          gradient_result.success)
        
        total_queries = (boundary_result.queries_used + 
                        confidence_result.queries_used + 
                        gradient_result.queries_used)
        
        return AttackResult(
            original_input=None,
            adversarial_input=comprehensive_info,
            original_prediction=None,
            adversarial_prediction=None,
            perturbation=None,
            success=overall_success,
            queries_used=total_queries,
            perturbation_norm=0.0,
            confidence_drop=0.0,
            attack_name=self.name,
            metadata={
                'attack_type': 'comprehensive',
                'target_info': target_info,
                'boundary_success': boundary_result.success,
                'confidence_success': confidence_result.success,
                'gradient_success': gradient_result.success,
                'sub_attacks': {
                    'boundary': boundary_result.metadata,
                    'confidence': confidence_result.metadata,
                    'gradient': gradient_result.metadata
                }
            }
        )
    
    def _generate_random_input(self) -> torch.Tensor:
        """Generate a random input for querying."""
        # Default: assume image-like input [1, 3, 32, 32]
        return torch.rand(1, 3, 32, 32)
    
    def _refine_boundary_point(self, uncertain_point: torch.Tensor) -> Optional[torch.Tensor]:
        """Refine a point to find exact decision boundary."""
        
        # Simple binary search approach
        original_pred = torch.argmax(self._predict(uncertain_point), dim=-1).item()
        
        # Try small perturbations to find class change
        for direction in [torch.randn_like(uncertain_point) for _ in range(10)]:
            direction = direction / torch.norm(direction)
            
            for step_size in [0.001, 0.01, 0.1]:
                test_point = uncertain_point + step_size * direction
                test_pred = torch.argmax(self._predict(test_point), dim=-1).item()
                
                if test_pred != original_pred:
                    # Found boundary crossing, refine with binary search
                    return self._binary_search_boundary(uncertain_point, test_point)
        
        return None
    
    def _binary_search_boundary(self, point_a: torch.Tensor, 
                               point_b: torch.Tensor) -> torch.Tensor:
        """Use binary search to find precise boundary point."""
        
        pred_a = torch.argmax(self._predict(point_a), dim=-1).item()
        pred_b = torch.argmax(self._predict(point_b), dim=-1).item()
        
        # Binary search
        for _ in range(10):  # Limit iterations
            mid_point = (point_a + point_b) / 2
            pred_mid = torch.argmax(self._predict(mid_point), dim=-1).item()
            
            if pred_mid == pred_a:
                point_a = mid_point
            else:
                point_b = mid_point
        
        return (point_a + point_b) / 2
    
    def _analyze_model_properties(self) -> Dict[str, Any]:
        """Analyze overall model properties from collected data."""
        
        if not self.query_log:
            return {}
        
        # Extract basic statistics
        all_responses = [log['response'] for log in self.query_log]
        all_confidences = [log['confidence'] for log in self.query_log]
        all_classes = [log['predicted_class'] for log in self.query_log]
        
        # Class distribution
        class_counts = defaultdict(int)
        for cls in all_classes:
            class_counts[cls] += 1
        
        num_classes = len(class_counts)
        class_balance = {k: v/len(all_classes) for k, v in class_counts.items()}
        
        # Confidence statistics
        confidence_stats = {
            'mean': np.mean(all_confidences),
            'std': np.std(all_confidences),
            'min': np.min(all_confidences),
            'max': np.max(all_confidences)
        }
        
        return {
            'estimated_classes': num_classes,
            'class_distribution': dict(class_counts),
            'class_balance': class_balance,
            'confidence_statistics': confidence_stats,
            'total_queries_analyzed': len(self.query_log)
        }
    
    def get_extracted_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of all extracted knowledge."""
        
        summary = {
            'total_queries': len(self.query_log),
            'extraction_methods_used': list(self.extracted_knowledge.keys()),
            'model_properties': self._analyze_model_properties()
        }
        
        # Add specific summaries for each extraction type
        for method, data in self.extracted_knowledge.items():
            if method == 'decision_boundaries':
                summary[f'{method}_summary'] = {
                    'classes_found': len(data['class_regions']),
                    'boundary_points_found': len(data['boundary_points']),
                    'uncertainty_regions': len(data['uncertainty_regions'])
                }
            elif method == 'confidence_patterns':
                summary[f'{method}_summary'] = {
                    'high_confidence_regions': len(data['high_confidence_regions']),
                    'low_confidence_regions': len(data['low_confidence_regions']),
                    'avg_confidence': np.mean(data['confidence_distribution'])
                }
            elif method == 'gradients':
                summary[f'{method}_summary'] = {
                    'gradient_samples': len(data['gradient_norms']),
                    'avg_gradient_norm': np.mean(data['gradient_norms']) if data['gradient_norms'] else 0,
                    'lipschitz_estimates': len(data['lipschitz_estimates'])
                }
        
        return summary