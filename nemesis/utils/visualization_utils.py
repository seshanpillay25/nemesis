"""
Visualization Utilities 🎨🔧

"Painting the divine canvas of battle and triumph"

Utilities for theming, color management, and
visual consistency across the Nemesis toolkit.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional, Any
import warnings

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ColorPalette:
    """
    Nemesis color palette with mythological theming.
    
    Provides consistent colors across all visualizations
    with both light and dark theme variants.
    """
    
    # Primary Nemesis colors
    NEMESIS_COLORS = {
        # Core brand colors
        'primary': '#667eea',        # Deep purple-blue
        'secondary': '#764ba2',      # Royal purple
        'accent': '#f093fb',         # Pink accent
        'gold': '#ffd700',           # Divine gold
        
        # Status colors
        'success': '#10b981',        # Emerald green
        'warning': '#f59e0b',        # Amber
        'danger': '#ef4444',         # Red
        'info': '#3b82f6',           # Blue
        
        # Battle colors
        'attack': '#dc2626',         # Battle red
        'defense': '#059669',        # Shield green
        'neutral': '#6b7280',        # Gray
        
        # Mythological palette
        'divine': '#ffd700',         # Gold
        'mortal': '#94a3b8',         # Silver-gray
        'darkness': '#1f2937',       # Dark
        'light': '#f9fafb',          # Light
        
        # Gradients base colors
        'gradient_start': '#667eea',
        'gradient_end': '#764ba2'
    }
    
    # Theme-specific palettes
    DARK_THEME = {
        'background': '#1a1a2e',
        'surface': '#16213e',
        'text_primary': '#ffffff',
        'text_secondary': '#e5e7eb',
        'border': '#374151',
        'accent_bg': '#312e81'
    }
    
    LIGHT_THEME = {
        'background': '#ffffff',
        'surface': '#f9fafb',
        'text_primary': '#111827',
        'text_secondary': '#6b7280',
        'border': '#d1d5db',
        'accent_bg': '#ede9fe'
    }
    
    @classmethod
    def get_color(cls, color_name: str, theme: str = 'dark') -> str:
        """Get color by name with theme consideration."""
        
        if color_name in cls.NEMESIS_COLORS:
            return cls.NEMESIS_COLORS[color_name]
        
        theme_colors = cls.DARK_THEME if theme == 'dark' else cls.LIGHT_THEME
        if color_name in theme_colors:
            return theme_colors[color_name]
        
        # Fallback to default
        return cls.NEMESIS_COLORS.get('primary', '#667eea')
    
    @classmethod
    def get_gradient_colors(cls, n_colors: int = 5) -> List[str]:
        """Get gradient colors for multi-series plots."""
        
        start_color = mcolors.hex2color(cls.NEMESIS_COLORS['gradient_start'])
        end_color = mcolors.hex2color(cls.NEMESIS_COLORS['gradient_end'])
        
        colors = []
        for i in range(n_colors):
            ratio = i / max(n_colors - 1, 1)
            color = [
                start_color[j] * (1 - ratio) + end_color[j] * ratio
                for j in range(3)
            ]
            colors.append(mcolors.rgb2hex(color))
        
        return colors
    
    @classmethod
    def get_attack_defense_colors(cls) -> Dict[str, str]:
        """Get colors for attack vs defense visualizations."""
        return {
            'attack': cls.NEMESIS_COLORS['attack'],
            'defense': cls.NEMESIS_COLORS['defense'],
            'neutral': cls.NEMESIS_COLORS['neutral']
        }
    
    @classmethod
    def get_severity_colors(cls) -> Dict[str, str]:
        """Get colors for severity levels."""
        return {
            'low': cls.NEMESIS_COLORS['success'],
            'medium': cls.NEMESIS_COLORS['warning'], 
            'high': cls.NEMESIS_COLORS['danger'],
            'critical': '#991b1b'  # Darker red
        }


class ThemeManager:
    """
    Theme management for consistent visual styling.
    
    Manages matplotlib, seaborn, and plotly themes
    with Nemesis branding.
    """
    
    def __init__(self, theme: str = 'nemesis_dark'):
        self.theme = theme
        self.color_palette = ColorPalette()
        self.current_theme_settings = {}
    
    def apply_matplotlib_theme(self):
        """Apply Nemesis theme to matplotlib."""
        
        if self.theme == 'nemesis_dark':
            theme_colors = self.color_palette.DARK_THEME
        else:
            theme_colors = self.color_palette.LIGHT_THEME
        
        # Set matplotlib style
        plt.style.use('dark_background' if 'dark' in self.theme else 'default')
        
        # Custom rcParams
        custom_params = {
            'figure.facecolor': theme_colors['background'],
            'axes.facecolor': theme_colors['surface'],
            'text.color': theme_colors['text_primary'],
            'axes.labelcolor': theme_colors['text_primary'],
            'xtick.color': theme_colors['text_primary'],
            'ytick.color': theme_colors['text_primary'],
            'axes.edgecolor': theme_colors['border'],
            'grid.color': theme_colors['border'],
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.prop_cycle': plt.cycler(
                'color', self.color_palette.get_gradient_colors(8)
            )
        }
        
        plt.rcParams.update(custom_params)
        self.current_theme_settings['matplotlib'] = custom_params
    
    def apply_seaborn_theme(self):
        """Apply Nemesis theme to seaborn."""
        
        if not SEABORN_AVAILABLE:
            warnings.warn("Seaborn not available, skipping seaborn theming")
            return
        
        # Set seaborn style
        if 'dark' in self.theme:
            sns.set_style("darkgrid")
        else:
            sns.set_style("whitegrid")
        
        # Set color palette
        nemesis_palette = self.color_palette.get_gradient_colors(10)
        sns.set_palette(nemesis_palette)
        
        self.current_theme_settings['seaborn'] = {
            'style': 'darkgrid' if 'dark' in self.theme else 'whitegrid',
            'palette': nemesis_palette
        }
    
    def get_plotly_theme(self) -> Dict[str, Any]:
        """Get Plotly theme configuration."""
        
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available, returning empty theme")
            return {}
        
        if 'dark' in self.theme:
            theme_colors = self.color_palette.DARK_THEME
        else:
            theme_colors = self.color_palette.LIGHT_THEME
        
        plotly_theme = {
            'layout': {
                'paper_bgcolor': theme_colors['background'],
                'plot_bgcolor': theme_colors['surface'],
                'font': {
                    'color': theme_colors['text_primary'],
                    'family': 'Arial, sans-serif',
                    'size': 12
                },
                'colorway': self.color_palette.get_gradient_colors(8),
                'title': {
                    'font': {'size': 16, 'color': theme_colors['text_primary']},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'gridcolor': theme_colors['border'],
                    'color': theme_colors['text_secondary']
                },
                'yaxis': {
                    'gridcolor': theme_colors['border'],
                    'color': theme_colors['text_secondary']
                },
                'legend': {
                    'bgcolor': theme_colors['surface'],
                    'bordercolor': theme_colors['border']
                }
            }
        }
        
        self.current_theme_settings['plotly'] = plotly_theme
        return plotly_theme
    
    def apply_all_themes(self):
        """Apply theme to all available visualization libraries."""
        self.apply_matplotlib_theme()
        self.apply_seaborn_theme()
        self.get_plotly_theme()  # Store for later use
    
    def reset_themes(self):
        """Reset all themes to defaults."""
        plt.rcdefaults()
        
        if SEABORN_AVAILABLE:
            sns.reset_defaults()
        
        self.current_theme_settings = {}


class PlotUtils:
    """
    Utility functions for common plotting operations.
    
    Provides high-level plotting functions with Nemesis theming
    and consistent styling.
    """
    
    def __init__(self, theme_manager: Optional[ThemeManager] = None):
        self.theme_manager = theme_manager or ThemeManager()
        self.color_palette = ColorPalette()
    
    def create_battle_progress_plot(self,
                                  rounds: List[int],
                                  strength_values: List[float],
                                  title: str = "Battle Progress",
                                  figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Create a battle progress visualization."""
        
        self.theme_manager.apply_matplotlib_theme()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot strength progression
        ax.plot(rounds, strength_values, 
               marker='o', linewidth=3, markersize=8,
               color=self.color_palette.get_color('primary'),
               label='Model Strength')
        
        # Fill area under curve
        ax.fill_between(rounds, strength_values, 
                       alpha=0.3, color=self.color_palette.get_color('primary'))
        
        # Styling
        ax.set_xlabel('Battle Round', fontweight='bold')
        ax.set_ylabel('Strength Level', fontweight='bold')
        ax.set_title(f'⚔️ {title}', fontsize=16, fontweight='bold', 
                    color=self.color_palette.get_color('gold'))
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_attack_effectiveness_chart(self,
                                        attack_names: List[str],
                                        success_rates: List[float],
                                        title: str = "Attack Effectiveness",
                                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Create attack effectiveness bar chart."""
        
        self.theme_manager.apply_matplotlib_theme()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bars with gradient colors
        colors = self.color_palette.get_gradient_colors(len(attack_names))
        bars = ax.bar(attack_names, success_rates, color=colors)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # Styling
        ax.set_ylabel('Success Rate', fontweight='bold')
        ax.set_title(f'🗡️ {title}', fontsize=16, fontweight='bold',
                    color=self.color_palette.get_color('gold'))
        ax.set_ylim(0, 1.1)
        
        # Rotate x-axis labels if needed
        if len(max(attack_names, key=len)) > 8:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def create_robustness_heatmap(self,
                                data: List[List[float]],
                                row_labels: List[str],
                                col_labels: List[str],
                                title: str = "Robustness Matrix",
                                figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Create robustness heatmap."""
        
        self.theme_manager.apply_matplotlib_theme()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        
        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f'{data[i][j]:.2f}',
                             ha="center", va="center", 
                             color="white" if data[i][j] < 0.5 else "black",
                             fontweight='bold')
        
        # Styling
        ax.set_title(f'🛡️ {title}', fontsize=16, fontweight='bold',
                    color=self.color_palette.get_color('gold'))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Robustness Score', rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def add_mythological_annotations(self, ax, annotations: List[Dict[str, Any]]):
        """Add mythological-themed annotations to plots."""
        
        mythological_symbols = {
            'victory': '🏆',
            'battle': '⚔️',
            'strength': '💪',
            'wisdom': '🦉',
            'lightning': '⚡',
            'shield': '🛡️',
            'crown': '👑',
            'fire': '🔥'
        }
        
        for annotation in annotations:
            symbol = mythological_symbols.get(annotation.get('type', ''), '')
            text = f"{symbol} {annotation.get('text', '')}"
            
            ax.annotate(text,
                       xy=annotation.get('xy', (0.5, 0.5)),
                       xytext=annotation.get('xytext', (0.6, 0.6)),
                       arrowprops=dict(arrowstyle='->', 
                                     color=self.color_palette.get_color('accent'),
                                     lw=2),
                       fontsize=annotation.get('fontsize', 12),
                       fontweight='bold',
                       color=self.color_palette.get_color('gold'))
    
    def save_plot_with_branding(self, 
                              fig: plt.Figure,
                              filename: str,
                              add_watermark: bool = True):
        """Save plot with Nemesis branding."""
        
        if add_watermark:
            # Add discrete watermark
            fig.text(0.99, 0.01, 'Generated by Nemesis 🏛️', 
                    fontsize=8, alpha=0.7, ha='right', va='bottom',
                    color=self.color_palette.get_color('text_secondary'))
        
        # Save with high quality
        fig.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor=fig.get_facecolor(), edgecolor='none')
    
    def create_legend_with_theme(self, 
                               labels: List[str],
                               colors: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create themed legend configuration."""
        
        if colors is None:
            colors = self.color_palette.get_gradient_colors(len(labels))
        
        legend_config = {
            'labels': labels,
            'colors': colors,
            'frameon': True,
            'fancybox': True,
            'shadow': True,
            'framealpha': 0.9,
            'facecolor': self.color_palette.get_color('surface'),
            'edgecolor': self.color_palette.get_color('border')
        }
        
        return legend_config