"""
I/O Utilities 📁🔧

"Sacred scrolls and divine record keeping"

Utilities for configuration management, results saving,
and battle logging with mythological theming.
"""

import json
import yaml
import pickle
import os
import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
import logging


class ConfigLoader:
    """
    Configuration loader for Nemesis settings.
    
    Supports YAML, JSON, and Python dict configurations
    with validation and default value handling.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        self.defaults = self._get_default_config()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default Nemesis configuration."""
        return {
            'nemesis': {
                'default_personality': 'adaptive',
                'battle_rounds': 10,
                'evolution_enabled': True,
                'attack_budget': 1000,
                'epsilon_range': [0.01, 0.1, 0.3]
            },
            'arena': {
                'name': 'Divine Colosseum',
                'auto_save': True,
                'replay_battles': True,
                'max_contestants': 100
            },
            'attacks': {
                'budget': 1000,
                'epsilon_range': [0.01, 0.3],
                'timeout': 300,
                'max_iterations': 100
            },
            'defenses': {
                'auto_evolve': True,
                'protection_level': 'medium',
                'efficiency_priority': 0.3
            },
            'logging': {
                'level': 'INFO',
                'save_logs': True,
                'log_dir': 'logs',
                'mythological_theme': True
            },
            'visualization': {
                'theme': 'nemesis_dark',
                'save_plots': True,
                'plot_dir': 'plots',
                'interactive': True
            }
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            self.config = self._load_yaml(config_path)
        elif config_path.suffix.lower() == '.json':
            self.config = self._load_json(config_path)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
        
        # Merge with defaults
        self.config = self._merge_with_defaults(self.config)
        return self.config
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise ValueError(f"Failed to load YAML config: {e}")
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON configuration."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load JSON config: {e}")
    
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration with defaults."""
        merged = self.defaults.copy()
        
        for section, values in config.items():
            if section in merged and isinstance(merged[section], dict):
                merged[section].update(values)
            else:
                merged[section] = values
        
        return merged
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def save_config(self, path: str, format: str = 'yaml') -> None:
        """Save current configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'yaml':
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


class ResultsSaver:
    """
    Results saving utility with multiple formats.
    
    Saves experiment results, battle outcomes, and metrics
    in various formats with proper organization.
    """
    
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = self._generate_session_id()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"nemesis_session_{timestamp}"
    
    def save_battle_results(self, 
                          results: Dict[str, Any],
                          experiment_name: str = "battle",
                          format: str = "json") -> str:
        """Save battle results with organized structure."""
        
        # Create experiment directory
        exp_dir = self.base_dir / self.session_id / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        results_with_meta = {
            'metadata': {
                'session_id': self.session_id,
                'experiment_name': experiment_name,
                'timestamp': datetime.datetime.now().isoformat(),
                'nemesis_version': '1.0.0'  # Could be dynamic
            },
            'results': results
        }
        
        # Save in requested format
        if format.lower() == 'json':
            file_path = exp_dir / "results.json"
            with open(file_path, 'w') as f:
                json.dump(results_with_meta, f, indent=2, default=str)
        
        elif format.lower() == 'yaml':
            file_path = exp_dir / "results.yaml"
            with open(file_path, 'w') as f:
                yaml.dump(results_with_meta, f, default_flow_style=False)
        
        elif format.lower() == 'pickle':
            file_path = exp_dir / "results.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(results_with_meta, f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(file_path)
    
    def save_model_analysis(self,
                          model_info: Dict[str, Any],
                          analysis_results: Dict[str, Any],
                          model_name: str = "analyzed_model") -> str:
        """Save model analysis results."""
        
        analysis_data = {
            'model_info': model_info,
            'analysis_results': analysis_results,
            'analysis_timestamp': datetime.datetime.now().isoformat()
        }
        
        return self.save_battle_results(
            analysis_data,
            experiment_name=f"model_analysis_{model_name}",
            format="json"
        )
    
    def save_metrics(self,
                    metrics: Dict[str, Any],
                    metrics_type: str = "general") -> str:
        """Save metrics data."""
        
        return self.save_battle_results(
            metrics,
            experiment_name=f"metrics_{metrics_type}",
            format="json"
        )
    
    def load_results(self, file_path: str) -> Dict[str, Any]:
        """Load previously saved results."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        
        elif file_path.suffix.lower() == '.pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def list_experiments(self) -> List[Dict[str, str]]:
        """List all experiments in the current session."""
        
        session_dir = self.base_dir / self.session_id
        if not session_dir.exists():
            return []
        
        experiments = []
        for exp_dir in session_dir.iterdir():
            if exp_dir.is_dir():
                # Look for result files
                result_files = list(exp_dir.glob("results.*"))
                if result_files:
                    experiments.append({
                        'name': exp_dir.name,
                        'path': str(result_files[0]),
                        'modified': datetime.datetime.fromtimestamp(
                            result_files[0].stat().st_mtime
                        ).isoformat()
                    })
        
        return experiments


class BattleLogger:
    """
    Mythologically-themed logger for Nemesis battles.
    
    Provides rich, themed logging with different verbosity levels
    and proper formatting for battle narratives.
    """
    
    def __init__(self, 
                 name: str = "Nemesis",
                 level: str = "INFO",
                 save_logs: bool = True,
                 log_dir: str = "logs"):
        
        self.name = name
        self.save_logs = save_logs
        self.log_dir = Path(log_dir)
        
        if save_logs:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup console handler with Rich
        console_handler = RichHandler(
            console=Console(),
            show_time=True,
            show_path=False,
            markup=True
        )
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # Custom formatter for mythological theme
        console_formatter = logging.Formatter(
            "[bold blue]%(name)s[/bold blue] | %(message)s",
            datefmt="[%X]"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Setup file handler if saving logs
        if save_logs:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"nemesis_battle_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # Save all to file
            
            file_formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def battle_start(self, nemesis_name: str, model_name: str):
        """Log battle start with epic formatting."""
        self.logger.info(
            f"🏛️ [bold gold]BATTLE COMMENCES[/bold gold] ⚔️\n"
            f"   Nemesis: [red]{nemesis_name}[/red]\n"
            f"   Champion: [blue]{model_name}[/blue]\n"
            f"   {'═' * 50}"
        )
    
    def battle_round(self, round_num: int, description: str):
        """Log battle round progress."""
        self.logger.info(f"⚔️ [bold yellow]Round {round_num}[/bold yellow]: {description}")
    
    def weakness_discovered(self, weakness_type: str, severity: str):
        """Log weakness discovery."""
        severity_colors = {
            'low': 'green',
            'medium': 'yellow', 
            'high': 'red',
            'critical': 'bold red'
        }
        color = severity_colors.get(severity.lower(), 'white')
        
        self.logger.warning(
            f"🔍 [bold red]WEAKNESS REVEALED[/bold red]: "
            f"[{color}]{weakness_type}[/{color}] "
            f"(Severity: [{color}]{severity}[/{color}])"
        )
    
    def defense_forged(self, defense_name: str, effectiveness: float):
        """Log defense creation."""
        self.logger.info(
            f"🛡️ [bold blue]ARMOR FORGED[/bold blue]: "
            f"[cyan]{defense_name}[/cyan] "
            f"(Effectiveness: [green]{effectiveness:.2%}[/green])"
        )
    
    def evolution_achieved(self, level: int, description: str):
        """Log model evolution."""
        self.logger.info(
            f"🌟 [bold purple]EVOLUTION ACHIEVED[/bold purple]: "
            f"Level {level} - {description}"
        )
    
    def battle_complete(self, 
                       victor: str,
                       final_strength: float,
                       battles_fought: int):
        """Log battle completion."""
        self.logger.info(
            f"🏆 [bold green]BATTLE COMPLETE[/bold green] 🏆\n"
            f"   Victor: [bold green]{victor}[/bold green]\n"
            f"   Final Strength: [green]{final_strength:.3f}[/green]\n"
            f"   Battles Fought: {battles_fought}\n"
            f"   {'═' * 50}"
        )
    
    def error_in_battle(self, error_type: str, description: str):
        """Log battle errors."""
        self.logger.error(
            f"💥 [bold red]DIVINE INTERVENTION[/bold red]: "
            f"{error_type} - {description}"
        )
    
    def attack_launched(self, attack_name: str, target: str, success: bool):
        """Log attack attempts."""
        if success:
            self.logger.info(
                f"⚡ [red]{attack_name}[/red] strikes [blue]{target}[/blue] - "
                f"[bold red]SUCCESS[/bold red]"
            )
        else:
            self.logger.info(
                f"⚡ [red]{attack_name}[/red] strikes [blue]{target}[/blue] - "
                f"[bold green]DEFENDED[/bold green]"
            )
    
    def metrics_updated(self, metric_name: str, value: float, change: Optional[float] = None):
        """Log metrics updates."""
        change_str = ""
        if change is not None:
            if change > 0:
                change_str = f" ([green]+{change:.3f}[/green])"
            elif change < 0:
                change_str = f" ([red]{change:.3f}[/red])"
        
        self.logger.info(
            f"📊 [cyan]{metric_name}[/cyan]: {value:.3f}{change_str}"
        )
    
    def save_battle_log(self, battle_data: Dict[str, Any], filename: Optional[str] = None):
        """Save structured battle log."""
        if not self.save_logs:
            return
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"battle_log_{timestamp}.json"
        
        log_path = self.log_dir / filename
        
        battle_log = {
            'metadata': {
                'logger_name': self.name,
                'timestamp': datetime.datetime.now().isoformat(),
                'log_level': self.logger.level
            },
            'battle_data': battle_data
        }
        
        with open(log_path, 'w') as f:
            json.dump(battle_log, f, indent=2, default=str)
        
        self.logger.info(f"📜 Battle log saved to [cyan]{log_path}[/cyan]")