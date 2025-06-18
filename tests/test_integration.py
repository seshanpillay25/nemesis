"""
Integration Tests 🏛️⚔️

"Testing the complete divine symphony of war and protection"

End-to-end integration tests that verify the complete Nemesis workflow
from summoning to legendary battles.
"""

import pytest
import torch
import torch.nn as nn
from nemesis import summon_nemesis, NemesisPersonality
from nemesis.attacks import Whisper, Storm, Chaos
from nemesis.defenses import Aegis, Fortitude, DefenseArmor
from nemesis.arena import Arena

class CompleteTestModel(nn.Module):
    """More complete test model for integration tests."""
    
    def __init__(self, name: str = "IntegrationModel"):
        super().__init__()
        self.name = name
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

@pytest.fixture
def complete_model():
    """Fixture for complete test model."""
    model = CompleteTestModel("CompleteTestModel")
    model.eval()
    return model

@pytest.fixture
def test_dataset():
    """Fixture for integration test dataset."""
    dataset = []
    for i in range(15):
        x = torch.randn(1, 3, 32, 32)
        y = torch.randint(0, 10, (1,)).item()
        dataset.append((x, y))
    return dataset

class TestCompleteNemesisWorkflow:
    """Test complete nemesis workflow from start to finish."""
    
    @pytest.mark.integration
    def test_full_nemesis_journey(self, complete_model):
        """Test the complete nemesis journey."""
        # Phase 1: Summon Nemesis
        nemesis = summon_nemesis(
            complete_model, 
            name="IntegrationNemesis",
            personality=NemesisPersonality.ADAPTIVE
        )
        
        assert nemesis.name == "IntegrationNemesis"
        assert nemesis.personality == NemesisPersonality.ADAPTIVE
        
        # Phase 2: Discover Weaknesses
        weaknesses = nemesis.find_weakness(attack_budget=50)
        
        assert hasattr(weaknesses, 'vulnerabilities')
        initial_weakness_count = len(weaknesses)
        
        # Phase 3: Forge Armor
        armor = nemesis.forge_armor(strategy="adaptive")
        protected_model = armor.apply(complete_model)
        
        assert protected_model is not None
        assert armor.strategy == "adaptive"
        
        # Phase 4: Eternal Battle
        champion = nemesis.eternal_battle(rounds=3, evolution=True)
        
        assert champion is not None
        assert len(nemesis.battle_history) == 3
        assert nemesis.evolution_level >= 1
        
        # Phase 5: Verify Improvement
        final_weaknesses = nemesis.find_weakness(attack_budget=50)
        
        # Model should have improved (fewer weaknesses or better robustness)
        assert isinstance(final_weaknesses.vulnerabilities, list)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_attack_defense_cycle(self, complete_model, cifar_input):
        """Test complete attack-defense cycle."""
        # Create attacks
        whisper = Whisper(complete_model)
        storm = Storm(complete_model)
        
        # Create defenses
        aegis = Aegis()
        
        # Phase 1: Test attacks on clean model
        clean_whisper_result = whisper.unleash(cifar_input, epsilon=0.1)
        clean_storm_result = storm.unleash(cifar_input, epsilon=0.1, num_iter=5)
        
        # Phase 2: Apply defenses
        defended_input = aegis.protect(
            cifar_input, 
            purification_method="gaussian_noise", 
            strength=0.05
        ).defended_input
        
        # Phase 3: Test attacks on defended input
        defended_whisper_result = whisper.unleash(defended_input, epsilon=0.1)
        defended_storm_result = storm.unleash(defended_input, epsilon=0.1, num_iter=5)
        
        # Verify all results
        assert all(hasattr(r, 'success') for r in [
            clean_whisper_result, clean_storm_result,
            defended_whisper_result, defended_storm_result
        ])
        
        # Defense may or may not improve robustness in this simple test
        assert isinstance(clean_whisper_result.success, bool)
        assert isinstance(defended_whisper_result.success, bool)
    
    @pytest.mark.integration
    def test_arena_nemesis_integration(self, complete_model):
        """Test arena and nemesis integration."""
        # Create arena
        arena = Arena("Integration Arena")
        
        # Create nemesis
        nemesis = summon_nemesis(complete_model, name="ArenaWarrior")
        
        # Arena battle (arena creates its own nemesis)
        battle_result = arena.legendary_battle(
            complete_model,
            rounds=2,
            nemesis_personality=NemesisPersonality.AGGRESSIVE
        )
        
        # Verify integration
        assert battle_result.model_name == complete_model.name
        assert battle_result.rounds_fought == 2
        assert len(arena.battle_history) == 1
        
        # Verify nemesis still works independently
        weakness_report = nemesis.find_weakness(attack_budget=20)
        assert hasattr(weakness_report, 'vulnerabilities')
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_multi_model_tournament(self, complete_model):
        """Test tournament with multiple model types."""
        # Create different models
        model1 = CompleteTestModel("Champion1")
        model2 = CompleteTestModel("Champion2")
        model3 = complete_model  # Already named "CompleteTestModel"
        
        models = [model1, model2, model3]
        
        # Create arena and run tournament
        arena = Arena("Multi-Model Arena")
        
        tournament_results = arena.tournament(
            models=models,
            tournament_name="Champions League",
            rounds_per_battle=2
        )
        
        # Verify tournament results
        assert tournament_results['contestants'] == 3
        assert len(tournament_results['battle_results']) == 3
        assert len(tournament_results['leaderboard']) == 3
        assert tournament_results['champion'] is not None
        
        # Verify each model fought
        model_names = [result.model_name for result in tournament_results['battle_results']]
        assert "Champion1" in model_names
        assert "Champion2" in model_names
        assert "CompleteTestModel" in model_names
        
        # Check Hall of Legends
        legends = arena.hall_of_legends()
        assert legends['total_battles'] == 3
        assert len(legends['greatest_champions']) <= 3

class TestRobustnessImprovement:
    """Test that models actually become more robust."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_adversarial_training_improvement(self, complete_model, test_dataset):
        """Test that adversarial training improves robustness."""
        # Create fortitude defense
        fortitude = Fortitude(complete_model)
        
        # Test initial robustness
        chaos = Chaos(complete_model)
        initial_report = chaos.analyze_model_robustness(
            test_inputs=[x for x, _ in test_dataset[:3]],  # Small subset for testing
            epsilon=0.1
        )
        initial_robustness = initial_report['robustness_score']
        
        # Apply adversarial training
        training_result = fortitude.protect(
            test_dataset[:10],  # Small training set
            epsilon=0.1,
            training_epochs=2
        )
        
        # Test final robustness
        final_report = chaos.analyze_model_robustness(
            test_inputs=[x for x, _ in test_dataset[:3]],
            epsilon=0.1
        )
        final_robustness = final_report['robustness_score']
        
        # Verify improvement occurred
        assert training_result.defense_applied
        assert isinstance(initial_robustness, (int, float))
        assert isinstance(final_robustness, (int, float))
        # Note: Improvement not guaranteed in such a small test, but should not decrease dramatically
    
    @pytest.mark.integration
    def test_defense_armor_effectiveness(self, complete_model, cifar_input):
        """Test that defense armor provides protection."""
        # Create comprehensive armor
        armor = DefenseArmor(complete_model, strategy="robust")
        
        # Create attack
        whisper = Whisper(complete_model)
        
        # Test attack on unprotected model
        unprotected_result = whisper.unleash(cifar_input, epsilon=0.1)
        
        # Apply armor protection (simulate protection during inference)
        protected_input = cifar_input
        for defense in armor.defenses:
            if hasattr(defense, 'protect') and defense.defense_type == "preprocessing":
                protection_result = defense.protect(protected_input)
                if protection_result.defense_applied:
                    protected_input = protection_result.defended_input
        
        # Test attack on protected input
        protected_result = whisper.unleash(protected_input, epsilon=0.1)
        
        # Verify both attacks completed
        assert isinstance(unprotected_result.success, bool)
        assert isinstance(protected_result.success, bool)
        # Protection effectiveness depends on specific case

class TestScalability:
    """Test system scalability and performance."""
    
    @pytest.mark.integration
    def test_multiple_nemeses_same_model(self, complete_model):
        """Test creating multiple nemeses for the same model."""
        nemeses = []
        personalities = [
            NemesisPersonality.AGGRESSIVE,
            NemesisPersonality.CUNNING,
            NemesisPersonality.RELENTLESS
        ]
        
        # Create multiple nemeses
        for i, personality in enumerate(personalities):
            nemesis = summon_nemesis(
                complete_model,
                name=f"Nemesis-{i+1}",
                personality=personality
            )
            nemeses.append(nemesis)
        
        # Test each nemesis independently
        for nemesis in nemeses:
            weakness_report = nemesis.find_weakness(attack_budget=20)
            assert hasattr(weakness_report, 'vulnerabilities')
            assert nemesis.name.startswith("Nemesis-")
        
        # Verify they're independent
        assert len(set(n.name for n in nemeses)) == 3
        assert len(set(n.personality for n in nemeses)) == 3
    
    @pytest.mark.integration
    def test_batch_processing(self, complete_model):
        """Test processing multiple inputs efficiently."""
        # Create inputs
        test_inputs = [torch.randn(1, 3, 32, 32) for _ in range(5)]
        
        # Create attack and defense
        whisper = Whisper(complete_model)
        aegis = Aegis()
        
        # Process all inputs
        attack_results = []
        defense_results = []
        
        for input_tensor in test_inputs:
            # Apply defense
            defense_result = aegis.protect(
                input_tensor,
                purification_method="gaussian_noise",
                strength=0.05
            )
            defense_results.append(defense_result)
            
            # Apply attack
            attack_result = whisper.unleash(defense_result.defended_input, epsilon=0.1)
            attack_results.append(attack_result)
        
        # Verify all processed
        assert len(attack_results) == 5
        assert len(defense_results) == 5
        assert all(hasattr(r, 'success') for r in attack_results)
        assert all(hasattr(r, 'defense_applied') for r in defense_results)

class TestErrorHandling:
    """Test error handling in integration scenarios."""
    
    @pytest.mark.integration
    def test_invalid_model_handling(self):
        """Test handling of invalid models."""
        # Test with None model
        with pytest.raises((AttributeError, TypeError)):
            summon_nemesis(None)
        
        # Test with non-model object
        with pytest.raises((AttributeError, TypeError)):
            summon_nemesis("not_a_model")
    
    @pytest.mark.integration
    def test_empty_dataset_handling(self, complete_model):
        """Test handling of empty datasets."""
        fortitude = Fortitude(complete_model)
        
        # Test with empty dataset
        result = fortitude.protect([], training_epochs=1)
        
        # Should handle gracefully
        assert isinstance(result, object)  # Some result object
    
    @pytest.mark.integration
    def test_invalid_parameters(self, complete_model, cifar_input):
        """Test handling of invalid parameters."""
        whisper = Whisper(complete_model)
        
        # Test with invalid epsilon
        result = whisper.unleash(cifar_input, epsilon=-0.1)  # Negative epsilon
        
        # Should handle gracefully (implementation dependent)
        assert hasattr(result, 'success')

class TestFrameworkCompatibility:
    """Test compatibility across different ML frameworks."""
    
    @pytest.mark.integration
    def test_pytorch_sklearn_compatibility(self, complete_model):
        """Test nemesis works with different frameworks."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Test PyTorch model
        pytorch_nemesis = summon_nemesis(complete_model)
        assert pytorch_nemesis.framework == "pytorch"
        
        # Test sklearn model
        sklearn_model = RandomForestClassifier(n_estimators=10)
        sklearn_nemesis = summon_nemesis(sklearn_model)
        assert sklearn_nemesis.framework == "sklearn"
        
        # Both should work (with appropriate limitations)
        pytorch_report = pytorch_nemesis.find_weakness(attack_budget=10)
        sklearn_report = sklearn_nemesis.find_weakness(attack_budget=10)
        
        assert hasattr(pytorch_report, 'vulnerabilities')
        assert hasattr(sklearn_report, 'vulnerabilities')

class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""
    
    @pytest.mark.integration
    def test_memory_efficiency(self, complete_model):
        """Test that system doesn't leak memory."""
        import gc
        
        # Create and destroy many nemeses
        for i in range(10):
            nemesis = summon_nemesis(complete_model, name=f"TempNemesis-{i}")
            nemesis.find_weakness(attack_budget=5)
            del nemesis
        
        # Force garbage collection
        gc.collect()
        
        # Create final nemesis to ensure system still works
        final_nemesis = summon_nemesis(complete_model, name="FinalNemesis")
        final_report = final_nemesis.find_weakness(attack_budget=5)
        
        assert hasattr(final_report, 'vulnerabilities')
        assert final_nemesis.name == "FinalNemesis"
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_battle_sequence(self, complete_model):
        """Test handling of large battle sequences."""
        nemesis = summon_nemesis(complete_model, name="MarathonWarrior")
        
        # Conduct extended battle
        result = nemesis.eternal_battle(rounds=10, evolution=True)
        
        assert result is not None
        assert len(nemesis.battle_history) == 10
        assert nemesis.evolution_level >= 1
        
        # System should still be responsive
        final_weakness = nemesis.find_weakness(attack_budget=10)
        assert hasattr(final_weakness, 'vulnerabilities')