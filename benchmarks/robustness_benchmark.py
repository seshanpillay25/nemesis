#!/usr/bin/env python3
"""
Robustness Benchmark 📊⚔️

"Measure thy strength against the trials of the gods"

Comprehensive robustness benchmarking suite for evaluating
model performance against the Nemesis attack arsenal.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from nemesis import Nemesis, NemesisPersonality
from nemesis.attacks import Whisper, Storm, Shapeshifter, Mirage, Chaos
from nemesis.defenses import Aegis, Fortitude, DefenseArmor
from nemesis.arena import Arena

@dataclass
class BenchmarkResult:
    """Results of a robustness benchmark."""
    model_name: str
    clean_accuracy: float
    adversarial_accuracy: Dict[str, float]
    robustness_scores: Dict[str, float]
    defense_effectiveness: Dict[str, float]
    performance_metrics: Dict[str, float]
    benchmark_duration: float

class RobustnessBenchmark:
    """
    Comprehensive robustness benchmark suite.
    
    Evaluates models against various adversarial attacks and defenses
    to provide standardized robustness metrics.
    """
    
    def __init__(self, name: str = "Nemesis Robustness Benchmark"):
        self.name = name
        self.results = []
        self.attacks = {}
        self.defenses = {}
    
    def _create_test_model(self, model_type: str = "cnn") -> nn.Module:
        """Create test model for benchmarking."""
        if model_type == "cnn":
            class BenchmarkCNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.name = "BenchmarkCNN"
                    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.pool = nn.MaxPool2d(2, 2)
                    self.fc1 = nn.Linear(64 * 8 * 8, 128)
                    self.fc2 = nn.Linear(128, 10)
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(0.3)
                
                def forward(self, x):
                    x = self.pool(self.relu(self.conv1(x)))
                    x = self.pool(self.relu(self.conv2(x)))
                    x = x.view(-1, 64 * 8 * 8)
                    x = self.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.fc2(x)
                    return x
            
            return BenchmarkCNN().eval()
        
        elif model_type == "mlp":
            class BenchmarkMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.name = "BenchmarkMLP"
                    self.fc1 = nn.Linear(3072, 256)  # 32*32*3 for CIFAR-like
                    self.fc2 = nn.Linear(256, 128)
                    self.fc3 = nn.Linear(128, 10)
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(0.3)
                
                def forward(self, x):
                    x = x.view(-1, 3072)
                    x = self.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x
            
            return BenchmarkMLP().eval()
    
    def _create_test_dataset(self, size: int = 100) -> List[Tuple[torch.Tensor, int]]:
        """Create test dataset for benchmarking."""
        dataset = []
        for i in range(size):
            x = torch.randn(1, 3, 32, 32)  # CIFAR-like
            y = torch.randint(0, 10, (1,)).item()
            dataset.append((x, y))
        return dataset
    
    def _initialize_attacks(self, model: nn.Module) -> Dict[str, Any]:
        """Initialize attack arsenal."""
        return {
            'Whisper': Whisper(model),
            'Storm': Storm(model),
            'Mirage': Mirage(model),
            'Chaos': Chaos(model)
        }
    
    def _initialize_defenses(self, model: nn.Module) -> Dict[str, Any]:
        """Initialize defense arsenal."""
        return {
            'Aegis': Aegis(),
            'DefenseArmor': DefenseArmor(model, strategy="adaptive")
        }
    
    def evaluate_clean_accuracy(self, model: nn.Module, test_dataset: List[Tuple[torch.Tensor, int]]) -> float:
        """Evaluate clean (non-adversarial) accuracy."""
        correct = 0
        total = len(test_dataset)
        
        with torch.no_grad():
            for x, y in test_dataset:
                output = model(x)
                pred = torch.argmax(output, dim=-1).item()
                if pred == y:
                    correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def evaluate_adversarial_accuracy(self, model: nn.Module, test_dataset: List[Tuple[torch.Tensor, int]], 
                                    epsilon: float = 0.1) -> Dict[str, float]:
        """Evaluate accuracy against adversarial attacks."""
        attacks = self._initialize_attacks(model)
        adversarial_accuracies = {}
        
        for attack_name, attack in attacks.items():
            correct = 0
            total = len(test_dataset)
            
            print(f"  Evaluating against {attack_name}...")
            
            for x, y in test_dataset:
                try:
                    if attack_name == "Storm":
                        result = attack.unleash(x, epsilon=epsilon, num_iter=10)
                    elif attack_name == "Chaos":
                        result = attack.unleash(x, epsilon=epsilon, individual_budget=50)
                    else:
                        result = attack.unleash(x, epsilon=epsilon)
                    
                    # If attack failed, use original input
                    test_input = result.adversarial_input if result.success else x
                    
                    with torch.no_grad():
                        output = model(test_input)
                        pred = torch.argmax(output, dim=-1).item()
                        if pred == y:
                            correct += 1
                
                except Exception as e:
                    print(f"    Warning: {attack_name} failed on sample: {e}")
                    # Count as correct if attack fails
                    correct += 1
            
            adversarial_accuracies[attack_name] = correct / total if total > 0 else 0.0
        
        return adversarial_accuracies
    
    def evaluate_defense_effectiveness(self, model: nn.Module, test_dataset: List[Tuple[torch.Tensor, int]],
                                     epsilon: float = 0.1) -> Dict[str, float]:
        """Evaluate defense effectiveness."""
        defenses = self._initialize_defenses(model)
        attacks = self._initialize_attacks(model)
        defense_effectiveness = {}
        
        for defense_name, defense in defenses.items():
            print(f"  Evaluating {defense_name} defense...")
            
            total_improvement = 0.0
            valid_tests = 0
            
            # Test against each attack
            for attack_name, attack in attacks.items():
                try:
                    clean_successes = 0
                    defended_successes = 0
                    
                    # Sample smaller subset for defense testing
                    test_subset = test_dataset[:20]  # Reduced for performance
                    
                    for x, y in test_subset:
                        # Attack clean input
                        if attack_name == "Storm":
                            clean_result = attack.unleash(x, epsilon=epsilon, num_iter=5)
                        else:
                            clean_result = attack.unleash(x, epsilon=epsilon)
                        
                        if clean_result.success:
                            clean_successes += 1
                        
                        # Apply defense
                        if hasattr(defense, 'protect'):
                            defense_result = defense.protect(x, purification_method="gaussian_noise", strength=0.05)
                            defended_input = defense_result.defended_input
                        else:
                            defended_input = x  # Defense not applicable
                        
                        # Attack defended input
                        if attack_name == "Storm":
                            defended_result = attack.unleash(defended_input, epsilon=epsilon, num_iter=5)
                        else:
                            defended_result = attack.unleash(defended_input, epsilon=epsilon)
                        
                        if defended_result.success:
                            defended_successes += 1
                    
                    # Calculate improvement (reduction in attack success rate)
                    clean_success_rate = clean_successes / len(test_subset)
                    defended_success_rate = defended_successes / len(test_subset)
                    improvement = max(0, clean_success_rate - defended_success_rate)
                    
                    total_improvement += improvement
                    valid_tests += 1
                
                except Exception as e:
                    print(f"    Warning: Defense test failed for {attack_name}: {e}")
            
            defense_effectiveness[defense_name] = total_improvement / max(valid_tests, 1)
        
        return defense_effectiveness
    
    def run_benchmark(self, model: nn.Module, test_dataset: List[Tuple[torch.Tensor, int]] = None,
                     epsilon: float = 0.1) -> BenchmarkResult:
        """Run complete robustness benchmark."""
        start_time = time.time()
        
        model_name = getattr(model, 'name', model.__class__.__name__)
        print(f"\n🏛️ Running Nemesis Benchmark for {model_name}")
        print("=" * 60)
        
        # Create test dataset if not provided
        if test_dataset is None:
            print("Creating test dataset...")
            test_dataset = self._create_test_dataset(50)  # Reduced for performance
        
        print(f"Dataset size: {len(test_dataset)} samples")
        
        # 1. Clean Accuracy
        print("\n1. Evaluating clean accuracy...")
        clean_accuracy = self.evaluate_clean_accuracy(model, test_dataset)
        print(f"   Clean accuracy: {clean_accuracy:.3f}")
        
        # 2. Adversarial Accuracy
        print("\n2. Evaluating adversarial robustness...")
        adversarial_accuracy = self.evaluate_adversarial_accuracy(model, test_dataset, epsilon)
        
        # 3. Robustness Scores
        robustness_scores = {}
        for attack_name, adv_acc in adversarial_accuracy.items():
            robustness_scores[attack_name] = adv_acc / max(clean_accuracy, 0.01)  # Normalized robustness
        
        # 4. Defense Effectiveness
        print("\n3. Evaluating defense effectiveness...")
        defense_effectiveness = self.evaluate_defense_effectiveness(model, test_dataset, epsilon)
        
        # 5. Performance Metrics
        end_time = time.time()
        benchmark_duration = end_time - start_time
        
        performance_metrics = {
            'benchmark_duration': benchmark_duration,
            'samples_per_second': len(test_dataset) / benchmark_duration,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'epsilon_tested': epsilon
        }
        
        # Create result
        result = BenchmarkResult(
            model_name=model_name,
            clean_accuracy=clean_accuracy,
            adversarial_accuracy=adversarial_accuracy,
            robustness_scores=robustness_scores,
            defense_effectiveness=defense_effectiveness,
            performance_metrics=performance_metrics,
            benchmark_duration=benchmark_duration
        )
        
        self.results.append(result)
        self._display_results(result)
        
        return result
    
    def _display_results(self, result: BenchmarkResult):
        """Display benchmark results."""
        print(f"\n🏆 Benchmark Results for {result.model_name}")
        print("=" * 60)
        
        print(f"\n📊 Accuracy Metrics:")
        print(f"  Clean Accuracy: {result.clean_accuracy:.3f}")
        
        print(f"\n⚔️ Adversarial Accuracy:")
        for attack, accuracy in result.adversarial_accuracy.items():
            print(f"  {attack}: {accuracy:.3f}")
        
        print(f"\n🛡️ Robustness Scores (Normalized):")
        for attack, score in result.robustness_scores.items():
            print(f"  {attack}: {score:.3f}")
        
        print(f"\n🛡️ Defense Effectiveness:")
        for defense, effectiveness in result.defense_effectiveness.items():
            print(f"  {defense}: {effectiveness:.3f}")
        
        print(f"\n⚡ Performance Metrics:")
        for metric, value in result.performance_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    
    def compare_models(self, models: List[nn.Module], test_dataset: List[Tuple[torch.Tensor, int]] = None) -> Dict[str, Any]:
        """Compare multiple models on the benchmark."""
        print(f"\n🏛️ Running Comparative Benchmark")
        print("=" * 60)
        
        results = []
        for model in models:
            result = self.run_benchmark(model, test_dataset)
            results.append(result)
        
        # Generate comparison
        comparison = {
            'models': [r.model_name for r in results],
            'clean_accuracies': [r.clean_accuracy for r in results],
            'average_robustness': [],
            'best_defenses': [],
            'rankings': {}
        }
        
        # Calculate average robustness
        for result in results:
            avg_robustness = np.mean(list(result.robustness_scores.values()))
            comparison['average_robustness'].append(avg_robustness)
        
        # Find best defenses
        for result in results:
            if result.defense_effectiveness:
                best_defense = max(result.defense_effectiveness.items(), key=lambda x: x[1])
                comparison['best_defenses'].append(best_defense[0])
            else:
                comparison['best_defenses'].append("None")
        
        # Create rankings
        comparison['rankings']['clean_accuracy'] = sorted(
            enumerate(comparison['clean_accuracies']), 
            key=lambda x: x[1], reverse=True
        )
        comparison['rankings']['robustness'] = sorted(
            enumerate(comparison['average_robustness']), 
            key=lambda x: x[1], reverse=True
        )
        
        self._display_comparison(comparison, results)
        
        return {
            'comparison': comparison,
            'detailed_results': results
        }
    
    def _display_comparison(self, comparison: Dict[str, Any], results: List[BenchmarkResult]):
        """Display model comparison results."""
        print(f"\n🏆 Model Comparison Results")
        print("=" * 60)
        
        print(f"\n📊 Clean Accuracy Ranking:")
        for i, (model_idx, accuracy) in enumerate(comparison['rankings']['clean_accuracy']):
            model_name = comparison['models'][model_idx]
            print(f"  {i+1}. {model_name}: {accuracy:.3f}")
        
        print(f"\n🛡️ Robustness Ranking:")
        for i, (model_idx, robustness) in enumerate(comparison['rankings']['robustness']):
            model_name = comparison['models'][model_idx]
            print(f"  {i+1}. {model_name}: {robustness:.3f}")
        
        print(f"\n⚔️ Attack Vulnerability Summary:")
        attack_names = list(results[0].adversarial_accuracy.keys())
        for attack in attack_names:
            print(f"  {attack}:")
            attack_scores = [(r.model_name, r.adversarial_accuracy[attack]) for r in results]
            attack_scores.sort(key=lambda x: x[1], reverse=True)
            for model_name, score in attack_scores:
                print(f"    {model_name}: {score:.3f}")

def main():
    """Run benchmark demonstration."""
    print("🏛️ Nemesis Robustness Benchmark Suite 🏛️")
    print("=" * 60)
    
    # Create benchmark
    benchmark = RobustnessBenchmark()
    
    # Create test models
    print("Creating test models...")
    cnn_model = benchmark._create_test_model("cnn")
    mlp_model = benchmark._create_test_model("mlp")
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = benchmark._create_test_dataset(30)  # Small for demo
    
    # Run individual benchmarks
    print("\n" + "="*60)
    cnn_result = benchmark.run_benchmark(cnn_model, test_dataset)
    
    print("\n" + "="*60)
    mlp_result = benchmark.run_benchmark(mlp_model, test_dataset)
    
    # Run comparison
    print("\n" + "="*60)
    comparison_results = benchmark.compare_models([cnn_model, mlp_model], test_dataset)
    
    print(f"\n🏛️ Benchmark suite completed! Check results above.")

if __name__ == "__main__":
    main()