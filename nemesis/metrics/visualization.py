"""
Metrics Visualization 📊🎨

"Painting the battlefield with data and insights"

Comprehensive visualization tools for metrics, battle progression,
and robustness analysis with mythological theming.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Some visualizations will be limited to matplotlib.")


class MetricsVisualizer:
    """
    Comprehensive visualization toolkit for Nemesis metrics.
    
    Creates beautiful, mythologically-themed visualizations
    for robustness analysis, battle progression, and performance metrics.
    """
    
    def __init__(self, style: str = "nemesis_dark"):
        self.style = style
        self._setup_matplotlib_style()
        
        # Nemesis color palette
        self.colors = {
            'primary': '#667eea',
            'secondary': '#764ba2', 
            'accent': '#f093fb',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'gold': '#ffd700',
            'silver': '#c0c0c0',
            'bronze': '#cd7f32'
        }
    
    def _setup_matplotlib_style(self):
        """Setup custom matplotlib style."""
        plt.style.use('dark_background')
        
        # Custom styling
        plt.rcParams.update({
            'figure.facecolor': '#1a1a2e',
            'axes.facecolor': '#16213e',
            'text.color': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 10
        })
    
    def plot_robustness_comparison(self, 
                                 robustness_scores: Dict[str, Dict[float, Any]],
                                 save_path: Optional[str] = None) -> None:
        """Create robustness comparison across attacks and epsilons."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🏛️ Adversarial Robustness Analysis', fontsize=16, color=self.colors['gold'])
        
        # Extract data
        attacks = list(robustness_scores.keys())
        epsilons = list(next(iter(robustness_scores.values())).keys())
        
        # 1. Overall Robustness Scores
        ax1 = axes[0, 0]
        overall_scores = []
        for attack in attacks:
            scores = [robustness_scores[attack][eps].overall_score for eps in epsilons]
            overall_scores.append(scores)
            ax1.plot(epsilons, scores, marker='o', label=attack, linewidth=2)
        
        ax1.set_xlabel('Epsilon (ε)')
        ax1.set_ylabel('Robustness Score')
        ax1.set_title('⚔️ Overall Robustness vs Perturbation Budget')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Attack Success Rates
        ax2 = axes[0, 1]
        for attack in attacks:
            success_rates = [robustness_scores[attack][eps].attack_success_rate for eps in epsilons]
            ax2.plot(epsilons, success_rates, marker='s', label=attack, linewidth=2)
        
        ax2.set_xlabel('Epsilon (ε)')
        ax2.set_ylabel('Attack Success Rate')
        ax2.set_title('🗡️ Attack Success Rate vs Epsilon')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Robustness Heatmap
        ax3 = axes[1, 0]
        heatmap_data = np.array(overall_scores)
        im = ax3.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
        ax3.set_xticks(range(len(epsilons)))
        ax3.set_xticklabels([f'{eps:.2f}' for eps in epsilons])
        ax3.set_yticks(range(len(attacks)))
        ax3.set_yticklabels(attacks)
        ax3.set_title('🔥 Robustness Heatmap')
        plt.colorbar(im, ax=ax3, label='Robustness Score')
        
        # 4. Perturbation Sensitivity
        ax4 = axes[1, 1]
        for attack in attacks:
            sensitivities = [robustness_scores[attack][eps].perturbation_sensitivity for eps in epsilons]
            ax4.plot(epsilons, sensitivities, marker='^', label=attack, linewidth=2)
        
        ax4.set_xlabel('Epsilon (ε)')
        ax4.set_ylabel('Perturbation Sensitivity')
        ax4.set_title('📊 Perturbation Sensitivity Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_battle_progression(self,
                              battle_history: List[Dict],
                              nemesis_name: str = "Unknown Nemesis",
                              save_path: Optional[str] = None) -> None:
        """Visualize battle progression and evolution."""
        
        if not battle_history:
            print("No battle history to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'⚔️ Battle Progression: {nemesis_name}', fontsize=16, color=self.colors['gold'])
        
        rounds = list(range(1, len(battle_history) + 1))
        
        # 1. Weaknesses Found Over Time
        ax1 = axes[0, 0]
        weaknesses = [b.get('weaknesses_found', 0) for b in battle_history]
        ax1.plot(rounds, weaknesses, marker='o', color=self.colors['danger'], linewidth=3)
        ax1.fill_between(rounds, weaknesses, alpha=0.3, color=self.colors['danger'])
        ax1.set_xlabel('Battle Round')
        ax1.set_ylabel('Weaknesses Found')
        ax1.set_title('🔍 Weakness Discovery')
        ax1.grid(True, alpha=0.3)
        
        # 2. Robustness Evolution
        ax2 = axes[0, 1]
        robustness = [b.get('robustness_score', 0) for b in battle_history]
        ax2.plot(rounds, robustness, marker='s', color=self.colors['success'], linewidth=3)
        ax2.fill_between(rounds, robustness, alpha=0.3, color=self.colors['success'])
        ax2.set_xlabel('Battle Round')
        ax2.set_ylabel('Robustness Score')
        ax2.set_title('💪 Strength Evolution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Evolution Level
        ax3 = axes[1, 0]
        evolution_levels = [b.get('evolution_level', 0) for b in battle_history]
        ax3.step(rounds, evolution_levels, where='post', color=self.colors['primary'], linewidth=3)
        ax3.fill_between(rounds, evolution_levels, step='post', alpha=0.3, color=self.colors['primary'])
        ax3.set_xlabel('Battle Round')
        ax3.set_ylabel('Evolution Level')
        ax3.set_title('🌟 Divine Evolution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Battle Intensity (Combined Metric)
        ax4 = axes[1, 1]
        if len(weaknesses) == len(robustness):
            # Battle intensity = weaknesses found / robustness (higher = more intense)
            intensity = [w / max(r, 0.1) for w, r in zip(weaknesses, robustness)]
            ax4.bar(rounds, intensity, color=self.colors['accent'], alpha=0.7)
            ax4.set_xlabel('Battle Round')
            ax4.set_ylabel('Battle Intensity')
            ax4.set_title('⚡ Battle Intensity')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_attack_effectiveness_radar(self,
                                      attack_analysis: Dict[str, Dict],
                                      save_path: Optional[str] = None) -> None:
        """Create radar chart for attack effectiveness comparison."""
        
        if not PLOTLY_AVAILABLE:
            self._plot_attack_effectiveness_bar(attack_analysis, save_path)
            return
        
        attacks = list(attack_analysis.keys())
        metrics = ['success_rate', 'query_efficiency', 'perturbation_magnitude']
        
        fig = go.Figure()
        
        colors = [self.colors['primary'], self.colors['accent'], self.colors['danger'], 
                 self.colors['success'], self.colors['warning']]
        
        for i, attack in enumerate(attacks):
            values = []
            for metric in metrics:
                if metric == 'success_rate':
                    val = attack_analysis[attack].get('success_rate', 0)
                elif metric == 'query_efficiency':
                    val = attack_analysis[attack].get('query_efficiency', {}).get('query_efficiency', 0)
                elif metric == 'perturbation_magnitude':
                    # Invert perturbation (lower is better)
                    mag = attack_analysis[attack].get('perturbation_magnitude', {}).get('mean', 1)
                    val = 1 / (1 + mag)  # Normalize
                else:
                    val = 0
                values.append(val)
            
            # Close the radar chart
            values.append(values[0])
            theta_labels = metrics + [metrics[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=theta_labels,
                fill='toself',
                name=attack,
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="🗡️ Attack Effectiveness Radar",
            font=dict(color='white'),
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#16213e'
        )
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
    
    def _plot_attack_effectiveness_bar(self,
                                     attack_analysis: Dict[str, Dict],
                                     save_path: Optional[str] = None) -> None:
        """Fallback bar chart for attack effectiveness."""
        
        attacks = list(attack_analysis.keys())
        success_rates = [attack_analysis[attack].get('success_rate', 0) for attack in attacks]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(attacks, success_rates, color=[self.colors['primary'], self.colors['accent'], 
                                                   self.colors['danger'], self.colors['success']])
        
        ax.set_ylabel('Success Rate')
        ax.set_title('🗡️ Attack Effectiveness Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rate:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_defense_effectiveness(self,
                                 defense_analysis: Dict[str, Dict],
                                 save_path: Optional[str] = None) -> None:
        """Visualize defense effectiveness across different attacks."""
        
        defenses = list(defense_analysis.keys())
        attacks = list(next(iter(defense_analysis.values())).keys())
        
        # Create effectiveness matrix
        effectiveness_matrix = []
        for defense in defenses:
            row = []
            for attack in attacks:
                eff = defense_analysis[defense][attack].get('defense_effectiveness', 0)
                row.append(eff)
            effectiveness_matrix.append(row)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(effectiveness_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(attacks)))
        ax.set_xticklabels(attacks, rotation=45, ha='right')
        ax.set_yticks(range(len(defenses)))
        ax.set_yticklabels(defenses)
        
        # Add text annotations
        for i in range(len(defenses)):
            for j in range(len(attacks)):
                text = ax.text(j, i, f'{effectiveness_matrix[i][j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('🛡️ Defense Effectiveness Matrix', pad=20)
        ax.set_xlabel('Attack Type')
        ax.set_ylabel('Defense Type')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Effectiveness Score', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_dashboard(self,
                                     evaluation_report: Dict,
                                     save_path: Optional[str] = None) -> None:
        """Create a comprehensive evaluation dashboard."""
        
        if not PLOTLY_AVAILABLE:
            print("Plotly required for interactive dashboard. Creating static plots...")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Model Performance', 'Robustness Trends', 
                          'Attack Success Rates', 'Defense Effectiveness',
                          'Evolution Progress', 'Recommendations'],
            specs=[[{"type": "indicator"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # 1. Performance Indicator
        clean_acc = evaluation_report.get('metrics', {}).get('clean_performance', {}).get('accuracy', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=clean_acc,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Clean Accuracy"},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': self.colors['primary']},
                   'steps': [{'range': [0, 0.6], 'color': self.colors['danger']},
                            {'range': [0.6, 0.8], 'color': self.colors['warning']},
                            {'range': [0.8, 1], 'color': self.colors['success']}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 0.9}}
        ), row=1, col=1)
        
        # Add more dashboard elements...
        
        fig.update_layout(
            title="🏛️ Nemesis Evaluation Dashboard",
            font=dict(color='white'),
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#16213e',
            height=1000
        )
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
    
    def save_battle_visualization_suite(self,
                                      battle_data: Dict,
                                      output_dir: str = "battle_visuals") -> None:
        """Save a complete suite of battle visualizations."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Battle progression
        if 'battle_history' in battle_data:
            self.plot_battle_progression(
                battle_data['battle_history'],
                battle_data.get('nemesis_name', 'Unknown'),
                f"{output_dir}/battle_progression.png"
            )
        
        # 2. Robustness analysis
        if 'robustness_scores' in battle_data:
            self.plot_robustness_comparison(
                battle_data['robustness_scores'],
                f"{output_dir}/robustness_analysis.png"
            )
        
        # 3. Attack effectiveness
        if 'attack_analysis' in battle_data:
            self.plot_attack_effectiveness_radar(
                battle_data['attack_analysis'],
                f"{output_dir}/attack_effectiveness.html"
            )
        
        # 4. Defense effectiveness
        if 'defense_analysis' in battle_data:
            self.plot_defense_effectiveness(
                battle_data['defense_analysis'],
                f"{output_dir}/defense_effectiveness.png"
            )
        
        print(f"📊 Battle visualization suite saved to {output_dir}/")