#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack Showcase Example

"Witness the full arsenal of mythological destruction"

This example demonstrates all the different attack types
available in the Nemesis arsenal.
"""

import torch
import torch.nn as nn
from nemesis.attacks import Whisper, Storm, Shapeshifter, Mirage, Chaos
from nemesis.attacks import Trojan, Corruption
from nemesis.attacks import MindThief, Oracle

class DemoModel(nn.Module):
    """Demo model for attack showcase."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def demonstrate_evasion_attacks(model, sample_input):
    """Demonstrate evasion attacks."""
    
    try:
        print("🌪️ EVASION ATTACKS - Tricks and Deceptions")
    except UnicodeEncodeError:
        print("EVASION ATTACKS - Tricks and Deceptions")
    print("-" * 50)
    
    # Whisper - FGSM
    try:
        print("\n🌬️ Whisper (FGSM) - Subtle perturbations")
    except UnicodeEncodeError:
        print("\nWhisper (FGSM) - Subtle perturbations")
    whisper = Whisper(model)
    whisper_result = whisper.unleash(sample_input, epsilon=0.1)
    print(f"   Success: {whisper_result.success}")
    print(f"   Perturbation norm: {whisper_result.perturbation_norm:.4f}")
    
    # Storm - PGD  
    try:
        print("\n⛈️ Storm (PGD) - Iterative devastation")
    except UnicodeEncodeError:
        print("\nStorm (PGD) - Iterative devastation")
    storm = Storm(model)
    storm_result = storm.unleash(sample_input, epsilon=0.1, num_iter=10)
    print(f"   Success: {storm_result.success}")
    print(f"   Perturbation norm: {storm_result.perturbation_norm:.4f}")
    
    # Shapeshifter - C&W
    try:
        print("\n🔮 Shapeshifter (C&W) - Optimized transformation")
    except UnicodeEncodeError:
        print("\nShapeshifter (C&W) - Optimized transformation")
    shapeshifter = Shapeshifter(model)
    shape_result = shapeshifter.unleash(sample_input, max_iterations=50)
    print(f"   Success: {shape_result.success}")
    print(f"   Perturbation norm: {shape_result.perturbation_norm:.4f}")
    
    # Mirage - DeepFool
    try:
        print("\n🏜️ Mirage (DeepFool) - Minimal boundary crossing")
    except UnicodeEncodeError:
        print("\nMirage (DeepFool) - Minimal boundary crossing")
    mirage = Mirage(model)
    mirage_result = mirage.unleash(sample_input)
    print(f"   Success: {mirage_result.success}")
    print(f"   Perturbation norm: {mirage_result.perturbation_norm:.4f}")
    
    # Chaos - AutoAttack ensemble
    try:
        print("\n🌪️ Chaos (AutoAttack) - Ensemble mayhem")
    except UnicodeEncodeError:
        print("\nChaos (AutoAttack) - Ensemble mayhem")
    chaos = Chaos(model)
    chaos_result = chaos.unleash(sample_input, epsilon=0.1)
    print(f"   Success: {chaos_result.success}")
    print(f"   Perturbation norm: {chaos_result.perturbation_norm:.4f}")

def demonstrate_poisoning_attacks(model):
    """Demonstrate poisoning attacks."""
    
    try:
        print("\n\n☠️ POISONING ATTACKS - Corruption from Within")
    except UnicodeEncodeError:
        print("\n\nPOISONING ATTACKS - Corruption from Within")
    print("-" * 50)
    
    # Create dummy dataset
    dummy_dataset = []
    for i in range(10):
        x = torch.randn(1, 3, 32, 32)
        y = torch.randint(0, 10, (1,)).item()
        dummy_dataset.append((x, y))
    
    # Trojan - Backdoor
    try:
        print("\n🐴 Trojan - Hidden backdoors")
    except UnicodeEncodeError:
        print("\nTrojan - Hidden backdoors")
    trojan = Trojan(model)
    sample_input = dummy_dataset[0][0]
    trojan_result = trojan.unleash(sample_input, target_class=5)
    print(f"   Trigger embedded: {trojan_result.success}")
    print(f"   Target class: {trojan_result.metadata.get('target_class', 'N/A')}")
    
    # Corruption - Label flipping
    try:
        print("\n🦠 Corruption - Dataset poisoning")
    except UnicodeEncodeError:
        print("\nCorruption - Dataset poisoning")
    corruption = Corruption(model)
    corrupt_result = corruption.unleash(
        dummy_dataset, 
        corruption_type="label_flip",
        corruption_rate=0.3
    )
    print(f"   Dataset corrupted: {corrupt_result.success}")
    print(f"   Samples affected: {corrupt_result.metadata.get('actual_corruptions', 0)}")

def demonstrate_extraction_attacks(model):
    """Demonstrate extraction attacks."""
    
    try:
        print("\n\n🔮 EXTRACTION ATTACKS - Stealing Secrets")
    except UnicodeEncodeError:
        print("\n\nEXTRACTION ATTACKS - Stealing Secrets")
    print("-" * 50)
    
    # Simple input generator for extraction
    def input_generator():
        return torch.randn(1, 3, 32, 32)
    
    # MindThief - Model extraction
    try:
        print("\n🧠 MindThief - Model stealing")
    except UnicodeEncodeError:
        print("\nMindThief - Model stealing")
    mindthief = MindThief(model)
    # Use small parameters for demo
    dummy_input = torch.randn(1, 3, 32, 32)
    theft_result = mindthief.unleash(
        dummy_input,
        input_generator=input_generator,
        num_queries=100,
        training_epochs=5
    )
    print(f"   Extraction successful: {theft_result.success}")
    print(f"   Model fidelity: {theft_result.metadata.get('model_fidelity', 'N/A')}")
    
    # Oracle - Query-based extraction
    try:
        print("\n🔮 Oracle - Information extraction")
    except UnicodeEncodeError:
        print("\nOracle - Information extraction")
    oracle = Oracle(model)
    oracle_result = oracle.unleash(
        dummy_input,
        attack_type="confidence",
        num_queries=50,
        input_generator=input_generator
    )
    print(f"   Information extracted: {oracle_result.success}")
    print(f"   High confidence regions: {oracle_result.metadata.get('high_confidence_regions', 0)}")

def main():
    """Run the attack showcase."""
    
    try:
        print("🗡️ Nemesis Attack Showcase 🗡️")
    except UnicodeEncodeError:
        print("Nemesis Attack Showcase")
    print("=" * 50)
    
    # Create demo model
    model = DemoModel()
    model.eval()
    
    # Create sample input
    sample_input = torch.randn(1, 3, 32, 32)
    
    try:
        print("🏛️ Preparing for demonstration of divine wrath...")
    except UnicodeEncodeError:
        print("Preparing for demonstration of divine wrath...")
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {sample_input.shape}")
    
    # Demonstrate each attack category
    demonstrate_evasion_attacks(model, sample_input)
    demonstrate_poisoning_attacks(model)
    demonstrate_extraction_attacks(model)
    
    try:
        print("\n\n🏛️ Attack showcase complete!")
    except UnicodeEncodeError:
        print("\n\nAttack showcase complete!")
    try:
        print("💫 Your model has witnessed the full arsenal of Nemesis.")
    except UnicodeEncodeError:
        print("Your model has witnessed the full arsenal of Nemesis.")
    try:
        print("⚔️ Use these attacks to forge stronger defenses!")
    except UnicodeEncodeError:
        print("Use these attacks to forge stronger defenses!")

if __name__ == "__main__":
    main()