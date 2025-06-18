#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arena Tournament Example

"Where champions are crowned and legends are born"

This example demonstrates the Arena system for hosting epic
tournaments between multiple models.
"""

import torch
import torch.nn as nn
from nemesis.arena import Arena
from nemesis import NemesisPersonality

class SimpleModel(nn.Module):
    """Simple model for tournament."""
    
    def __init__(self, name: str, hidden_size: int = 128):
        super().__init__()
        self.name = name
        self.fc1 = nn.Linear(784, hidden_size)  # MNIST style
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def create_tournament_models():
    """Create a diverse set of models for tournament."""
    
    models = [
        SimpleModel("Spartan", hidden_size=256),
        SimpleModel("Gladiator", hidden_size=128), 
        SimpleModel("Centurion", hidden_size=64),
        SimpleModel("Legionnaire", hidden_size=32)
    ]
    
    # Set all models to eval mode
    for model in models:
        model.eval()
    
    return models

def main():
    """Run the arena tournament example."""
    
    try:
        print("🏛️ Arena Tournament Example 🏛️")
    except UnicodeEncodeError:
        print("Arena Tournament Example")
    print("=" * 50)
    
    # Create the sacred arena
    arena = Arena("Colosseum of Champions")
    
    # Create tournament contestants
    models = create_tournament_models()
    
    try:
        print(f"\n🏆 Tournament contestants:")
    except UnicodeEncodeError:
        print(f"\nTournament contestants:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model.name}")
    
    try:
        print("\n⚔️ Beginning tournament battles...")
    except UnicodeEncodeError:
        print("\nBeginning tournament battles...")
    
    # Host the tournament
    tournament_results = arena.tournament(
        models=models,
        tournament_name="Battle of the Architectures",
        rounds_per_battle=3
    )
    
    try:
        print(f"\n🏆 Tournament Results:")
    except UnicodeEncodeError:
        print(f"\nTournament Results:")
    print(f"Champion: {tournament_results['champion']['model_name']}")
    print(f"Robustness Score: {tournament_results['champion']['robustness_score']:.3f}")
    
    # Individual battles for demonstration
    print("\n⚔️ Individual Battle Demonstrations:")
    
    # Aggressive nemesis battle
    print("\n🔥 Battle vs Aggressive Nemesis:")
    aggressive_battle = arena.legendary_battle(
        model=models[0],
        rounds=3,
        nemesis_personality=NemesisPersonality.AGGRESSIVE,
        battle_name="Spartan vs Fury"
    )
    
    # Cunning nemesis battle  
    print("\n🎭 Battle vs Cunning Nemesis:")
    cunning_battle = arena.legendary_battle(
        model=models[1], 
        rounds=3,
        nemesis_personality=NemesisPersonality.CUNNING,
        battle_name="Gladiator vs Shadow"
    )
    
    # Display Hall of Legends
    print("\n🏛️ Consulting the Hall of Legends...")
    legends = arena.hall_of_legends()
    
    # Save arena state
    arena.save_arena_state("tournament_results.json")
    
    print("\n🏛️ Tournament complete! Check tournament_results.json for detailed results.")

if __name__ == "__main__":
    main()