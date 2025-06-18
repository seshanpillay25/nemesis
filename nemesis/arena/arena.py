"""
Arena - The Sacred Battleground 🏛️

"Where legends are born and champions are forged"

The main Arena class where models face their nemesis in epic confrontations.
Every battle is recorded, every victory celebrated, every defeat learned from.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn

from .. import Nemesis, NemesisPersonality
from ..attacks import *
from ..defenses import *

try:
    # Try to create console with UTF-8 support
    console = Console(force_terminal=True, legacy_windows=False)
except Exception:
    # Fallback for Windows compatibility
    console = Console(force_terminal=True, legacy_windows=True, width=100)

@dataclass
class BattleResult:
    """Result of an epic battle between model and nemesis."""
    
    model_name: str
    nemesis_name: str
    rounds_fought: int
    model_victories: int
    nemesis_victories: int
    draws: int
    final_robustness_score: float
    improvement_gained: float
    battle_duration: float
    legendary_moments: List[Dict[str, Any]]
    battle_id: str

class Arena:
    """
    The Sacred Arena - Where Models Face Their Nemesis
    
    Like the ancient colosseum where gladiators proved their worth,
    the Arena is where AI models are tested, challenged, and ultimately
    transformed into champions.
    """
    
    def __init__(self, name: str = "Arena of Legends"):
        """
        Initialize the Arena.
        
        Args:
            name: Name of this arena
        """
        self.name = name
        self.battle_history = []
        self.active_battles = {}
        self.champion_models = {}
        self.legendary_artifacts = []
        
        try:
            console.print(Panel(
                f"[bold yellow]{self.name}[/bold yellow]\n"
                "[italic]Where legends are born and champions are forged[/italic]",
                title="[bold red]ARENA INITIALIZED[/bold red]",
                border_style="yellow"
            ))
        except (UnicodeEncodeError, UnicodeError) as e:
            # Fallback for encoding issues
            try:
                import sys
                if hasattr(sys.stdout, 'reconfigure'):
                    sys.stdout.reconfigure(encoding='utf-8')
                print(f"Arena {self.name} initialized")
            except Exception:
                # Final fallback - plain text
                print(f"Arena {self.name} initialized")
        except Exception as e:
            # Final fallback - plain text
            print(f"Arena {self.name} initialized")
    
    def summon_nemesis(self, model: Any, name: Optional[str] = None,
                      personality: str = NemesisPersonality.ADAPTIVE) -> Nemesis:
        """
        Summon a personalized nemesis for a model.
        
        Args:
            model: The model to face its nemesis
            name: Custom name for the nemesis
            personality: Nemesis personality
            
        Returns:
            Summoned Nemesis ready for battle
        """
        nemesis = Nemesis(model, name=name, personality=personality)
        
        try:
            console.print(f"[bold red]Nemesis {nemesis.name} has been summoned![/bold red]")
        except (UnicodeEncodeError, UnicodeError):
            print(f"Nemesis {nemesis.name} has been summoned!")
        
        return nemesis
    
    def legendary_battle(self, model: Any, rounds: int = 10,
                        nemesis_personality: str = NemesisPersonality.ADAPTIVE,
                        evolution_enabled: bool = True,
                        battle_name: Optional[str] = None) -> BattleResult:
        """
        Host a legendary battle between model and nemesis.
        
        Args:
            model: Model to battle
            rounds: Number of battle rounds
            nemesis_personality: Personality of the nemesis
            evolution_enabled: Whether model can evolve during battle
            battle_name: Custom battle name
            
        Returns:
            BattleResult containing the epic outcome
        """
        import time
        import uuid
        
        battle_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Summon nemesis
        nemesis = self.summon_nemesis(model, personality=nemesis_personality)
        
        # Initialize battle tracking
        model_victories = 0
        nemesis_victories = 0
        draws = 0
        legendary_moments = []
        
        model_name = getattr(model, 'name', f'Model-{id(model)}')
        battle_title = battle_name or f"Epic Battle: {model_name} vs {nemesis.name}"
        
        try:
            console.print(Panel(
                f"[bold yellow]{battle_title}[/bold yellow]\n"
                f"Rounds: {rounds}\n"
                f"Nemesis Personality: {nemesis_personality}\n"
                f"Evolution: {'Enabled' if evolution_enabled else 'Disabled'}",
                title="[bold red]LEGENDARY BATTLE BEGINS[/bold red]",
                border_style="red"
            ))
        except (UnicodeEncodeError, UnicodeError):
            print(f"Legendary Battle: {battle_title}\nRounds: {rounds}\nPersonality: {nemesis_personality}")
        
        # Battle progress bar
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            
            battle_task = progress.add_task("Battle Progress", total=rounds)
            
            # Battle rounds
            for round_num in range(1, rounds + 1):
                try:
                    console.print(f"\n[bold cyan]Round {round_num}/{rounds}[/bold cyan]")
                except (UnicodeEncodeError, UnicodeError):
                    print(f"\nRound {round_num}/{rounds}")
                
                # Round 1: Nemesis attacks
                weaknesses = nemesis.find_weakness()
                
                if len(weaknesses) > 0:
                    try:
                        console.print(f"Nemesis discovered {len(weaknesses)} weaknesses!")
                    except (UnicodeEncodeError, UnicodeError):
                        print(f"Nemesis discovered {len(weaknesses)} weaknesses!")
                    nemesis_victories += 1
                    
                    # Record legendary moment
                    if len(weaknesses) >= 3:
                        legendary_moments.append({
                            'round': round_num,
                            'type': 'devastating_discovery',
                            'description': f'Nemesis uncovered {len(weaknesses)} critical weaknesses',
                            'weakness_count': len(weaknesses)
                        })
                else:
                    try:
                        console.print("Model stood strong against all attacks!")
                    except (UnicodeEncodeError, UnicodeError):
                        print("Model stood strong against all attacks!")
                    model_victories += 1
                    
                    legendary_moments.append({
                        'round': round_num,
                        'type': 'heroic_defense',
                        'description': 'Model deflected all nemesis attacks',
                        'round_outcome': 'model_victory'
                    })
                
                # Round 2: Model adapts (if evolution enabled)
                if evolution_enabled and len(weaknesses) > 0:
                    armor = nemesis.forge_armor()
                    evolved_model = armor.apply(model)
                    
                    # Test if evolution was successful
                    post_evolution_weaknesses = nemesis.find_weakness()
                    
                    if len(post_evolution_weaknesses) < len(weaknesses):
                        improvement = len(weaknesses) - len(post_evolution_weaknesses)
                        try:
                            console.print(f"Model evolved! Reduced vulnerabilities by {improvement}")
                        except (UnicodeEncodeError, UnicodeError):
                            print(f"Model evolved! Reduced vulnerabilities by {improvement}")
                        
                        if improvement >= 2:
                            legendary_moments.append({
                                'round': round_num,
                                'type': 'divine_evolution',
                                'description': f'Model underwent divine transformation, healing {improvement} weaknesses',
                                'improvement': improvement
                            })
                
                # Update progress
                progress.update(battle_task, advance=1)
                
                # Brief pause for dramatic effect
                time.sleep(0.1)
        
        battle_duration = time.time() - start_time
        
        # Calculate final scores
        total_battles = model_victories + nemesis_victories + draws
        if total_battles > 0:
            final_robustness = model_victories / total_battles
        else:
            final_robustness = 0.5
        
        # Estimate improvement (simplified)
        improvement_gained = max(0.0, final_robustness - 0.5)
        
        # Create battle result
        battle_result = BattleResult(
            model_name=model_name,
            nemesis_name=nemesis.name,
            rounds_fought=rounds,
            model_victories=model_victories,
            nemesis_victories=nemesis_victories,
            draws=draws,
            final_robustness_score=final_robustness,
            improvement_gained=improvement_gained,
            battle_duration=battle_duration,
            legendary_moments=legendary_moments,
            battle_id=battle_id
        )
        
        # Record battle
        self.battle_history.append(battle_result)
        
        # Display final results
        self._display_battle_results(battle_result)
        
        return battle_result
    
    def tournament(self, models: List[Any], tournament_name: str = "Arena Tournament",
                  rounds_per_battle: int = 5) -> Dict[str, Any]:
        """
        Host a tournament between multiple models.
        
        Args:
            models: List of models to compete
            tournament_name: Name of the tournament
            rounds_per_battle: Rounds per individual battle
            
        Returns:
            Tournament results
        """
        try:
            console.print(Panel(
                f"[bold yellow]{tournament_name}[/bold yellow]\n"
                f"Contestants: {len(models)}\n"
                f"Rounds per Battle: {rounds_per_battle}",
                title="[bold red]TOURNAMENT BEGINS[/bold red]",
                border_style="yellow"
            ))
        except (UnicodeEncodeError, UnicodeError):
            print(f"Tournament {tournament_name} begins with {len(models)} contestants")
        
        tournament_results = {
            'tournament_name': tournament_name,
            'contestants': len(models),
            'battle_results': [],
            'leaderboard': [],
            'champion': None
        }
        
        # Battle each model
        for i, model in enumerate(models):
            model_name = getattr(model, 'name', f'Contestant-{i+1}')
            try:
                console.print(f"\n[bold cyan]{model_name} enters the arena![/bold cyan]")
            except (UnicodeEncodeError, UnicodeError):
                print(f"\n{model_name} enters the arena!")
            
            battle_result = self.legendary_battle(
                model, 
                rounds=rounds_per_battle,
                battle_name=f"{tournament_name}: {model_name}'s Trial"
            )
            
            tournament_results['battle_results'].append(battle_result)
            
            # Add to leaderboard
            tournament_results['leaderboard'].append({
                'model_name': model_name,
                'robustness_score': battle_result.final_robustness_score,
                'improvement': battle_result.improvement_gained,
                'victories': battle_result.model_victories,
                'legendary_moments': len(battle_result.legendary_moments)
            })
        
        # Sort leaderboard by robustness score
        tournament_results['leaderboard'].sort(
            key=lambda x: x['robustness_score'], reverse=True
        )
        
        # Crown champion
        if tournament_results['leaderboard']:
            champion = tournament_results['leaderboard'][0]
            tournament_results['champion'] = champion
            
            console.print(Panel(
                f"👑 [bold yellow]CHAMPION: {champion['model_name']}[/bold yellow] 👑\n"
                f"🛡️ Robustness Score: {champion['robustness_score']:.3f}\n"
                f"⚡ Improvement: {champion['improvement']:.3f}\n"
                f"⭐ Legendary Moments: {champion['legendary_moments']}",
                title="[bold red]TOURNAMENT CHAMPION[/bold red]",
                border_style="yellow"
            ))
        
        return tournament_results
    
    def hall_of_legends(self) -> Dict[str, Any]:
        """
        Display the Hall of Legends with battle statistics.
        
        Returns:
            Hall of Legends data
        """
        if not self.battle_history:
            console.print("🏛️ [italic]The Hall of Legends awaits its first champions...[/italic]")
            return {'message': 'No battles recorded yet'}
        
        # Compile legends
        legends = {
            'total_battles': len(self.battle_history),
            'greatest_champions': [],
            'most_legendary_moments': [],
            'epic_statistics': {}
        }
        
        # Find greatest champions
        champions = {}
        for battle in self.battle_history:
            model_name = battle.model_name
            if model_name not in champions:
                champions[model_name] = {
                    'battles': 0,
                    'total_robustness': 0.0,
                    'total_improvement': 0.0,
                    'legendary_moments': 0
                }
            
            champions[model_name]['battles'] += 1
            champions[model_name]['total_robustness'] += battle.final_robustness_score
            champions[model_name]['total_improvement'] += battle.improvement_gained
            champions[model_name]['legendary_moments'] += len(battle.legendary_moments)
        
        # Calculate averages and sort
        for name, stats in champions.items():
            stats['avg_robustness'] = stats['total_robustness'] / stats['battles']
            stats['avg_improvement'] = stats['total_improvement'] / stats['battles']
        
        legends['greatest_champions'] = sorted(
            [{'name': name, **stats} for name, stats in champions.items()],
            key=lambda x: x['avg_robustness'],
            reverse=True
        )[:5]  # Top 5
        
        # Find most legendary moments
        all_moments = []
        for battle in self.battle_history:
            for moment in battle.legendary_moments:
                moment['battle_id'] = battle.battle_id
                moment['model_name'] = battle.model_name
                all_moments.append(moment)
        
        legends['most_legendary_moments'] = sorted(
            all_moments,
            key=lambda x: x.get('improvement', x.get('weakness_count', 0)),
            reverse=True
        )[:3]  # Top 3 moments
        
        # Epic statistics
        total_rounds = sum(b.rounds_fought for b in self.battle_history)
        total_victories = sum(b.model_victories for b in self.battle_history)
        total_duration = sum(b.battle_duration for b in self.battle_history)
        
        legends['epic_statistics'] = {
            'total_rounds_fought': total_rounds,
            'total_model_victories': total_victories,
            'total_battle_time': total_duration,
            'average_battle_duration': total_duration / len(self.battle_history),
            'overall_model_win_rate': total_victories / total_rounds if total_rounds > 0 else 0
        }
        
        # Display Hall of Legends
        self._display_hall_of_legends(legends)
        
        return legends
    
    def _display_battle_results(self, battle: BattleResult):
        """Display the results of an epic battle."""
        
        # Determine winner
        if battle.model_victories > battle.nemesis_victories:
            winner = "MODEL VICTORIOUS!"
            winner_color = "green"
        elif battle.nemesis_victories > battle.model_victories:
            winner = "NEMESIS TRIUMPHANT!"
            winner_color = "red"
        else:
            winner = "EPIC DRAW!"
            winner_color = "yellow"
        
        # Create results table
        table = Table(title=f"Battle Results: {battle.battle_id}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")
        
        table.add_row("Model Name", battle.model_name)
        table.add_row("Nemesis Name", battle.nemesis_name)
        table.add_row("Rounds Fought", str(battle.rounds_fought))
        table.add_row("Model Victories", str(battle.model_victories))
        table.add_row("Nemesis Victories", str(battle.nemesis_victories))
        table.add_row("Final Robustness", f"{battle.final_robustness_score:.3f}")
        table.add_row("Improvement Gained", f"{battle.improvement_gained:.3f}")
        table.add_row("Legendary Moments", str(len(battle.legendary_moments)))
        table.add_row("Battle Duration", f"{battle.battle_duration:.2f}s")
        
        console.print(table)
        
        console.print(Panel(
            f"[bold {winner_color}]{winner}[/bold {winner_color}]",
            title="[bold red]BATTLE CONCLUSION[/bold red]",
            border_style=winner_color
        ))
        
        # Display legendary moments
        if battle.legendary_moments:
            console.print("\n⭐ [bold yellow]Legendary Moments:[/bold yellow]")
            for moment in battle.legendary_moments:
                console.print(f"  🏛️ Round {moment['round']}: {moment['description']}")
    
    def _display_hall_of_legends(self, legends: Dict[str, Any]):
        """Display the Hall of Legends."""
        
        console.print(Panel(
            "🏛️ [bold yellow]HALL OF LEGENDS[/bold yellow] 🏛️\n"
            "[italic]Where the greatest champions are remembered for eternity[/italic]",
            border_style="yellow"
        ))
        
        # Champions table
        if legends['greatest_champions']:
            champions_table = Table(title="🏆 Greatest Champions")
            champions_table.add_column("Champion", style="yellow")
            champions_table.add_column("Battles", style="cyan")
            champions_table.add_column("Avg Robustness", style="green")
            champions_table.add_column("Avg Improvement", style="blue")
            champions_table.add_column("Legendary Moments", style="red")
            
            for champion in legends['greatest_champions']:
                champions_table.add_row(
                    champion['name'],
                    str(champion['battles']),
                    f"{champion['avg_robustness']:.3f}",
                    f"{champion['avg_improvement']:.3f}",
                    str(champion['legendary_moments'])
                )
            
            console.print(champions_table)
        
        # Epic statistics
        stats = legends['epic_statistics']
        console.print("\n📊 [bold blue]Epic Statistics:[/bold blue]")
        console.print(f"  ⚔️ Total Battles: {legends['total_battles']}")
        console.print(f"  🎯 Total Rounds: {stats['total_rounds_fought']}")
        console.print(f"  🏆 Model Victories: {stats['total_model_victories']}")
        console.print(f"  ⏱️ Total Battle Time: {stats['total_battle_time']:.2f}s")
        console.print(f"  📈 Model Win Rate: {stats['overall_model_win_rate']:.3f}")
    
    def save_arena_state(self, filepath: str):
        """Save arena state to file."""
        import json
        import datetime
        
        arena_data = {
            'name': self.name,
            'saved_at': datetime.datetime.now().isoformat(),
            'total_battles': len(self.battle_history),
            'battle_summary': [
                {
                    'battle_id': b.battle_id,
                    'model_name': b.model_name,
                    'nemesis_name': b.nemesis_name,
                    'final_robustness': b.final_robustness_score,
                    'legendary_moments': len(b.legendary_moments)
                }
                for b in self.battle_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(arena_data, f, indent=2)
        
        console.print(f"🏛️ Arena state saved to {filepath}")
    
    def get_battle_by_id(self, battle_id: str) -> Optional[BattleResult]:
        """Get battle result by ID."""
        for battle in self.battle_history:
            if battle.battle_id == battle_id:
                return battle
        return None