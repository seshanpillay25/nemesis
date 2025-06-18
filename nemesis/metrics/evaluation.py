"""
Model Evaluation 🏛️📊

"The arena where truth is revealed through battle"

Comprehensive evaluation framework for models, attacks, and defenses
with detailed analysis and reporting capabilities.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table
from rich.progress import track

from .robustness import RobustnessMetrics, AttackMetrics, DefenseMetrics


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    model_name: str
    evaluation_type: str
    timestamp: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    robustness_scores: Dict[float, Any] = field(default_factory=dict)
    attack_analysis: Dict[str, Any] = field(default_factory=dict)
    defense_analysis: Dict[str, Any] = field(default_factory=dict)
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class ModelEvaluator:
    """
    Comprehensive model evaluation framework.
    
    Evaluates models across multiple dimensions including
    clean performance, adversarial robustness, and efficiency.
    """
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.robustness_metrics = RobustnessMetrics()
        self.attack_metrics = AttackMetrics()
        self.defense_metrics = DefenseMetrics()
    
    def evaluate_model_comprehensive(self,
                                   model: torch.nn.Module,
                                   test_dataset: List[Tuple[torch.Tensor, int]],
                                   attacks: List[Callable],
                                   defenses: Optional[List[Callable]] = None,
                                   epsilon_values: Optional[List[float]] = None) -> EvaluationReport:
        """Run comprehensive model evaluation."""
        
        start_time = time.time()
        model_name = getattr(model, 'name', model.__class__.__name__)
        
        self.console.print(f"🏛️ [bold blue]Comprehensive Evaluation: {model_name}[/bold blue]")
        self.console.print("=" * 60)
        
        # Initialize report
        report = EvaluationReport(
            model_name=model_name,
            evaluation_type="comprehensive",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # 1. Clean Performance Evaluation
        self.console.print("\n📊 [bold]Clean Performance Evaluation[/bold]")
        clean_metrics = self._evaluate_clean_performance(model, test_dataset)
        report.metrics['clean_performance'] = clean_metrics
        
        # 2. Adversarial Robustness Evaluation
        self.console.print("\n⚔️ [bold]Adversarial Robustness Evaluation[/bold]")
        robustness_results = self._evaluate_adversarial_robustness(
            model, test_dataset, attacks, epsilon_values
        )
        report.robustness_scores = robustness_results
        
        # 3. Attack Analysis
        self.console.print("\n🗡️ [bold]Attack Effectiveness Analysis[/bold]")
        attack_analysis = self._analyze_attack_effectiveness(
            model, test_dataset, attacks
        )
        report.attack_analysis = attack_analysis
        
        # 4. Defense Evaluation (if provided)
        if defenses:
            self.console.print("\n🛡️ [bold]Defense Effectiveness Evaluation[/bold]")
            defense_analysis = self._evaluate_defense_effectiveness(
                model, test_dataset, attacks, defenses
            )
            report.defense_analysis = defense_analysis
        
        # 5. Performance Statistics
        end_time = time.time()
        report.performance_stats = {
            'evaluation_duration': end_time - start_time,
            'samples_evaluated': len(test_dataset),
            'attacks_tested': len(attacks),
            'defenses_tested': len(defenses) if defenses else 0
        }
        
        # 6. Generate Recommendations
        report.recommendations = self._generate_recommendations(report)
        
        # Display summary
        self._display_evaluation_summary(report)
        
        return report
    
    def _evaluate_clean_performance(self, 
                                  model: torch.nn.Module,
                                  test_dataset: List[Tuple[torch.Tensor, int]]) -> Dict[str, float]:
        """Evaluate clean model performance."""
        
        model.eval()
        correct = 0
        total = 0
        inference_times = []
        confidence_scores = []
        
        for x, y in track(test_dataset, description="Evaluating clean performance..."):
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(x)
                pred = torch.argmax(outputs, dim=-1).item()
                confidence = torch.softmax(outputs, dim=-1).max().item()
            
            end_time = time.time()
            
            correct += (pred == y)
            total += 1
            inference_times.append(end_time - start_time)
            confidence_scores.append(confidence)
        
        return {
            'accuracy': correct / total,
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'mean_confidence': np.mean(confidence_scores),
            'std_confidence': np.std(confidence_scores)
        }
    
    def _evaluate_adversarial_robustness(self,
                                       model: torch.nn.Module,
                                       test_dataset: List[Tuple[torch.Tensor, int]],
                                       attacks: List[Callable],
                                       epsilon_values: Optional[List[float]]) -> Dict[str, Any]:
        """Evaluate adversarial robustness across attacks and epsilons."""
        
        if epsilon_values is None:
            epsilon_values = [0.01, 0.03, 0.1, 0.3]
        
        results = {}
        
        for attack in attacks:
            attack_name = getattr(attack, 'name', attack.__class__.__name__)
            self.console.print(f"  Testing {attack_name}...")
            
            # Evaluate across epsilon values
            epsilon_results = self.robustness_metrics.evaluate_epsilon_robustness(
                model, test_dataset[:20], attack.unleash, epsilon_values  # Subset for speed
            )
            results[attack_name] = epsilon_results
        
        return results
    
    def _analyze_attack_effectiveness(self,
                                    model: torch.nn.Module,
                                    test_dataset: List[Tuple[torch.Tensor, int]],
                                    attacks: List[Callable]) -> Dict[str, Any]:
        """Analyze effectiveness of different attacks."""
        
        analysis = {}
        
        for attack in attacks:
            attack_name = getattr(attack, 'name', attack.__class__.__name__)
            self.console.print(f"  Analyzing {attack_name}...")
            
            # Run attacks
            attack_results = []
            original_inputs = []
            adversarial_inputs = []
            
            for x, y in test_dataset[:15]:  # Subset for analysis
                try:
                    result = attack.unleash(x, epsilon=0.1)
                    attack_results.append(result)
                    original_inputs.append(x)
                    adversarial_inputs.append(result.adversarial_input if result.success else x)
                except Exception as e:
                    # Create dummy result for failed attacks
                    class DummyResult:
                        success = False
                        queries_used = 0
                    attack_results.append(DummyResult())
                    original_inputs.append(x)
                    adversarial_inputs.append(x)
            
            # Calculate metrics
            success_rate = self.attack_metrics.calculate_attack_success_rate(attack_results)
            perturbation_stats = self.attack_metrics.calculate_perturbation_magnitude(
                original_inputs, adversarial_inputs
            )
            query_stats = self.attack_metrics.calculate_query_efficiency(attack_results)
            
            analysis[attack_name] = {
                'success_rate': success_rate,
                'perturbation_magnitude': perturbation_stats,
                'query_efficiency': query_stats
            }
        
        return analysis
    
    def _evaluate_defense_effectiveness(self,
                                      model: torch.nn.Module,
                                      test_dataset: List[Tuple[torch.Tensor, int]],
                                      attacks: List[Callable],
                                      defenses: List[Callable]) -> Dict[str, Any]:
        """Evaluate effectiveness of defenses."""
        
        defense_analysis = {}
        
        for defense in defenses:
            defense_name = getattr(defense, 'name', defense.__class__.__name__)
            self.console.print(f"  Testing {defense_name}...")
            
            defense_results = {}
            
            for attack in attacks:
                attack_name = getattr(attack, 'name', attack.__class__.__name__)
                
                # Attack without defense
                clean_results = []
                for x, y in test_dataset[:10]:  # Small subset
                    try:
                        result = attack.unleash(x, epsilon=0.1)
                        clean_results.append(result)
                    except:
                        class DummyResult:
                            success = False
                        clean_results.append(DummyResult())
                
                # Attack with defense
                defended_results = []
                for x, y in test_dataset[:10]:
                    try:
                        # Apply defense
                        if hasattr(defense, 'protect'):
                            defense_result = defense.protect(x)
                            defended_input = defense_result.defended_input
                        else:
                            defended_input = x
                        
                        # Attack defended input
                        result = attack.unleash(defended_input, epsilon=0.1)
                        defended_results.append(result)
                    except:
                        class DummyResult:
                            success = False
                        defended_results.append(DummyResult())
                
                # Calculate effectiveness
                effectiveness = self.defense_metrics.calculate_defense_effectiveness(
                    clean_results, defended_results
                )
                defense_results[attack_name] = effectiveness
            
            defense_analysis[defense_name] = defense_results
        
        return defense_analysis
    
    def _generate_recommendations(self, report: EvaluationReport) -> List[str]:
        """Generate actionable recommendations based on evaluation."""
        
        recommendations = []
        
        # Clean performance recommendations
        clean_acc = report.metrics.get('clean_performance', {}).get('accuracy', 0)
        if clean_acc < 0.8:
            recommendations.append("🎯 Consider improving base model performance before adversarial training")
        
        # Robustness recommendations
        if report.robustness_scores:
            avg_robustness = []
            for attack_results in report.robustness_scores.values():
                for epsilon_result in attack_results.values():
                    avg_robustness.append(epsilon_result.overall_score)
            
            if avg_robustness and np.mean(avg_robustness) < 0.5:
                recommendations.append("⚔️ Model shows low adversarial robustness - consider adversarial training")
        
        # Attack-specific recommendations
        if report.attack_analysis:
            high_success_attacks = []
            for attack_name, analysis in report.attack_analysis.items():
                if analysis.get('success_rate', 0) > 0.8:
                    high_success_attacks.append(attack_name)
            
            if high_success_attacks:
                recommendations.append(f"🛡️ Highly vulnerable to: {', '.join(high_success_attacks)} - prioritize defenses")
        
        # General recommendations
        recommendations.extend([
            "📈 Implement gradual epsilon training for better robustness",
            "🔄 Consider ensemble methods for improved defense",
            "📊 Regular robustness monitoring recommended"
        ])
        
        return recommendations
    
    def _display_evaluation_summary(self, report: EvaluationReport):
        """Display comprehensive evaluation summary."""
        
        self.console.print(f"\n🏆 [bold green]Evaluation Summary: {report.model_name}[/bold green]")
        self.console.print("=" * 60)
        
        # Create summary table
        table = Table(title="Model Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")
        
        # Clean performance
        clean_acc = report.metrics.get('clean_performance', {}).get('accuracy', 0)
        status = "✅ Good" if clean_acc > 0.8 else "⚠️ Needs Improvement"
        table.add_row("Clean Accuracy", f"{clean_acc:.3f}", status)
        
        # Average robustness
        if report.robustness_scores:
            robustness_scores = []
            for attack_results in report.robustness_scores.values():
                for epsilon_result in attack_results.values():
                    robustness_scores.append(epsilon_result.overall_score)
            
            avg_robustness = np.mean(robustness_scores) if robustness_scores else 0
            status = "✅ Robust" if avg_robustness > 0.6 else "⚠️ Vulnerable"
            table.add_row("Avg Robustness", f"{avg_robustness:.3f}", status)
        
        self.console.print(table)
        
        # Recommendations
        if report.recommendations:
            self.console.print(f"\n💡 [bold yellow]Recommendations:[/bold yellow]")
            for rec in report.recommendations:
                self.console.print(f"  • {rec}")


class BattleAnalyzer:
    """
    Specialized analyzer for battle scenarios and evolution tracking.
    """
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
    
    def analyze_battle_progression(self, battle_history: List[Dict]) -> Dict[str, Any]:
        """Analyze how model performance evolved through battles."""
        
        if not battle_history:
            return {'error': 'No battle history available'}
        
        # Extract progression metrics
        rounds = len(battle_history)
        weakness_progression = [b.get('weaknesses_found', 0) for b in battle_history]
        robustness_progression = [b.get('robustness_score', 0) for b in battle_history]
        evolution_levels = [b.get('evolution_level', 0) for b in battle_history]
        
        # Calculate trends
        weakness_trend = np.polyfit(range(rounds), weakness_progression, 1)[0] if rounds > 1 else 0
        robustness_trend = np.polyfit(range(rounds), robustness_progression, 1)[0] if rounds > 1 else 0
        
        return {
            'total_battles': rounds,
            'weakness_reduction_rate': -weakness_trend,  # Negative because fewer weaknesses is better
            'robustness_improvement_rate': robustness_trend,
            'final_evolution_level': evolution_levels[-1] if evolution_levels else 0,
            'improvement_detected': robustness_trend > 0,
            'battle_efficiency': self._calculate_battle_efficiency(battle_history)
        }
    
    def _calculate_battle_efficiency(self, battle_history: List[Dict]) -> float:
        """Calculate how efficiently battles improved the model."""
        
        if len(battle_history) < 2:
            return 0.0
        
        initial_robustness = battle_history[0].get('robustness_score', 0)
        final_robustness = battle_history[-1].get('robustness_score', 0)
        
        improvement = final_robustness - initial_robustness
        battles_count = len(battle_history)
        
        # Efficiency = improvement per battle
        efficiency = improvement / battles_count if battles_count > 0 else 0
        return max(0, efficiency)  # Only positive efficiency
    
    def generate_battle_report(self, 
                             nemesis_name: str,
                             battle_history: List[Dict],
                             final_metrics: Dict) -> str:
        """Generate a narrative battle report."""
        
        analysis = self.analyze_battle_progression(battle_history)
        
        report = f"""
🏛️ BATTLE REPORT: {nemesis_name.upper()} 🏛️

Battle Statistics:
• Total Battles Fought: {analysis.get('total_battles', 0)}
• Evolution Level Achieved: {analysis.get('final_evolution_level', 0)}
• Robustness Improvement Rate: {analysis.get('robustness_improvement_rate', 0):.4f}/battle

Performance Progression:
• Initial Weaknesses: {battle_history[0].get('weaknesses_found', 'Unknown') if battle_history else 'N/A'}
• Final Weaknesses: {battle_history[-1].get('weaknesses_found', 'Unknown') if battle_history else 'N/A'}
• Battle Efficiency: {analysis.get('battle_efficiency', 0):.4f}

Final Assessment:
"""
        
        if analysis.get('improvement_detected', False):
            report += "✅ Model successfully evolved through adversarial trials\n"
            report += "🏆 Nemesis fulfilled its divine purpose\n"
        else:
            report += "⚠️ Limited improvement detected - consider extended training\n"
            report += "🔄 Additional battles may be required\n"
        
        report += f"\n💪 Current Robustness Score: {final_metrics.get('robustness_score', 'Unknown')}"
        report += f"\n⚔️ Ready for greater challenges: {'Yes' if analysis.get('improvement_detected') else 'Needs more training'}"
        
        return report