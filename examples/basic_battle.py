#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Battle Example

"Every legend begins with a single battle"

This example demonstrates the core Nemesis functionality:
summoning a nemesis, discovering weaknesses, and forging defenses.
"""

import torch
import torch.nn as nn
from nemesis import summon_nemesis, NemesisPersonality

# Create a simple model for demonstration
class SimpleModel(nn.Module):
    """A simple CNN for CIFAR-10 style inputs."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    """Run the basic battle example."""
    
    try:
        print("🏛️ Nemesis Basic Battle Example 🏛️")
    except UnicodeEncodeError:
        print("Nemesis Basic Battle Example")
    print("=" * 50)
    
    # Create model
    model = SimpleModel()
    model.eval()
    
    # Create sample input (CIFAR-10 style: 32x32 RGB)
    sample_input = torch.randn(1, 3, 32, 32)
    
    try:
        print("\n⚡ Summoning your nemesis...")
    except UnicodeEncodeError:
        print("\nSummoning your nemesis...")
    
    # Summon nemesis with adaptive personality
    nemesis = summon_nemesis(
        model, 
        name="AdaptiveBane",
        personality=NemesisPersonality.ADAPTIVE
    )
    
    try:
        print(f"🎭 Nemesis {nemesis.name} awakens with {nemesis.personality} personality!")
    except UnicodeEncodeError:
        print(f"Nemesis {nemesis.name} awakens with {nemesis.personality} personality!")
    
    # Test different attack personalities
    personalities = [
        NemesisPersonality.AGGRESSIVE,
        NemesisPersonality.CUNNING, 
        NemesisPersonality.RELENTLESS
    ]
    
    for personality in personalities:
        try:
            print(f"\n🗡️ Testing {personality} attacks...")
        except UnicodeEncodeError:
            print(f"\nTesting {personality} attacks...")
        
        # Create nemesis with specific personality
        test_nemesis = summon_nemesis(model, personality=personality)
        
        # Find weaknesses
        weaknesses = test_nemesis.find_weakness(attack_budget=100)
        
        try:
            print(f"💀 Found {len(weaknesses)} vulnerabilities")
        except UnicodeEncodeError:
            print(f"Found {len(weaknesses)} vulnerabilities")
        for vuln in weaknesses.vulnerabilities:
            print(f"  - {vuln['attack_type']}: {vuln['success_rate']:.2f} success rate")
    
    try:
        print("\n🔨 Forging armor against discovered weaknesses...")
    except UnicodeEncodeError:
        print("\nForging armor against discovered weaknesses...")
    
    # Forge defensive armor
    armor = nemesis.forge_armor(strategy="adaptive")
    protected_model = armor.apply(model)
    
    try:
        print("🛡️ Armor forged successfully!")
    except UnicodeEncodeError:
        print("Armor forged successfully!")
    
    print("\n⚔️ Beginning eternal battle...")
    
    # Conduct eternal battle for model improvement
    champion_model = nemesis.eternal_battle(rounds=5, evolution=True)
    
    print(f"🏆 Battle complete! Model evolution level: {nemesis.evolution_level}")
    
    # Display battle statistics
    stats = nemesis.get_statistics()
    print(f"\n📊 Battle Statistics:")
    print(f"  - Battles fought: {len(nemesis.battle_history)}")
    print(f"  - Evolution level: {nemesis.evolution_level}")
    print(f"  - Victories: {nemesis.victories}")
    print(f"  - Defeats: {nemesis.defeats}")
    
    print("\n🏛️ Example complete! Your model has faced its nemesis and grown stronger.")

if __name__ == "__main__":
    main()