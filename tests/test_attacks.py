"""
Test Attack Arsenal 🗡️

"Testing the weapons of divine retribution"

Tests for all adversarial attacks in the Nemesis arsenal.
"""

import pytest
import torch
import numpy as np
from nemesis.attacks import (
    Whisper, Storm, Shapeshifter, Mirage, Chaos,
    Trojan, Corruption,
    MindThief, Oracle,
    AttackBase, AttackResult
)

class TestAttackBase:
    """Test the base attack class."""
    
    def test_attack_result_creation(self):
        """Test AttackResult dataclass creation."""
        result = AttackResult(
            original_input=torch.randn(1, 3, 32, 32),
            adversarial_input=torch.randn(1, 3, 32, 32),
            original_prediction=torch.randn(1, 10),
            adversarial_prediction=torch.randn(1, 10),
            perturbation=torch.randn(1, 3, 32, 32),
            success=True,
            queries_used=100,
            perturbation_norm=0.1,
            confidence_drop=0.2,
            attack_name="TestAttack",
            metadata={"test": "data"}
        )
        
        assert result.success == True
        assert result.queries_used == 100
        assert result.attack_name == "TestAttack"
        assert result.metadata["test"] == "data"

class TestEvasionAttacks:
    """Test evasion attacks."""
    
    def test_whisper_attack_creation(self, cnn_model):
        """Test Whisper (FGSM) attack creation."""
        whisper = Whisper(cnn_model)
        
        assert whisper.name == "Whisper"
        assert whisper.model is cnn_model
        assert whisper.framework == "pytorch"
    
    def test_whisper_attack_execution(self, cnn_model, cifar_input):
        """Test Whisper attack execution."""
        whisper = Whisper(cnn_model)
        
        result = whisper.unleash(cifar_input, epsilon=0.1)
        
        assert isinstance(result, AttackResult)
        assert result.attack_name == "Whisper"
        assert result.original_input is not None
        assert result.adversarial_input is not None
        assert result.perturbation is not None
        assert isinstance(result.success, bool)
        assert result.queries_used >= 0
    
    def test_storm_attack_creation(self, cnn_model):
        """Test Storm (PGD) attack creation."""
        storm = Storm(cnn_model)
        
        assert storm.name == "Storm"
        assert storm.model is cnn_model
    
    def test_storm_attack_execution(self, cnn_model, cifar_input):
        """Test Storm attack execution."""
        storm = Storm(cnn_model)
        
        result = storm.unleash(
            cifar_input, 
            epsilon=0.1, 
            num_iter=5,  # Reduced for testing
            alpha=0.01
        )
        
        assert isinstance(result, AttackResult)
        assert result.attack_name == "Storm"
        assert result.queries_used >= 0
    
    def test_shapeshifter_attack_creation(self, cnn_model):
        """Test Shapeshifter (C&W) attack creation."""
        shapeshifter = Shapeshifter(cnn_model)
        
        assert shapeshifter.name == "Shapeshifter"
        assert shapeshifter.model is cnn_model
    
    @pytest.mark.slow
    def test_shapeshifter_attack_execution(self, cnn_model, cifar_input):
        """Test Shapeshifter attack execution."""
        shapeshifter = Shapeshifter(cnn_model)
        
        result = shapeshifter.unleash(
            cifar_input,
            max_iterations=10,  # Reduced for testing
            c=1.0
        )
        
        assert isinstance(result, AttackResult)
        assert result.attack_name == "Shapeshifter"
    
    def test_mirage_attack_creation(self, cnn_model):
        """Test Mirage (DeepFool) attack creation."""
        mirage = Mirage(cnn_model)
        
        assert mirage.name == "Mirage"
        assert mirage.model is cnn_model
    
    def test_mirage_attack_execution(self, cnn_model, cifar_input):
        """Test Mirage attack execution."""
        mirage = Mirage(cnn_model)
        
        result = mirage.unleash(
            cifar_input,
            max_iterations=10  # Reduced for testing
        )
        
        assert isinstance(result, AttackResult)
        assert result.attack_name == "Mirage"
    
    def test_chaos_attack_creation(self, cnn_model):
        """Test Chaos (AutoAttack) creation."""
        chaos = Chaos(cnn_model)
        
        assert chaos.name == "Chaos"
        assert chaos.model is cnn_model
        assert hasattr(chaos, 'whisper')
        assert hasattr(chaos, 'storm')
        assert hasattr(chaos, 'shapeshifter')
        assert hasattr(chaos, 'mirage')
    
    @pytest.mark.slow
    def test_chaos_attack_execution(self, cnn_model, cifar_input):
        """Test Chaos ensemble attack execution."""
        chaos = Chaos(cnn_model)
        
        result = chaos.unleash(
            cifar_input,
            epsilon=0.1,
            individual_budget=10  # Reduced for testing
        )
        
        assert isinstance(result, AttackResult)
        assert result.attack_name == "Chaos"
        assert 'attack_sequence' in result.metadata or 'successful_attack' in result.metadata
    
    def test_attack_parameters(self, cnn_model, cifar_input):
        """Test various attack parameters."""
        whisper = Whisper(cnn_model)
        
        # Test different epsilon values
        for epsilon in [0.01, 0.05, 0.1]:
            result = whisper.unleash(cifar_input, epsilon=epsilon)
            assert result.metadata['epsilon'] == epsilon
        
        # Test targeted vs untargeted
        result_untargeted = whisper.unleash(cifar_input, targeted=False)
        result_targeted = whisper.unleash(cifar_input, targeted=True, target_class=5)
        
        assert result_untargeted.metadata['targeted'] == False
        assert result_targeted.metadata['targeted'] == True
        assert result_targeted.metadata['target_class'] == 5

class TestPoisoningAttacks:
    """Test poisoning attacks."""
    
    def test_trojan_attack_creation(self, cnn_model):
        """Test Trojan attack creation."""
        trojan = Trojan(cnn_model)
        
        assert trojan.name == "Trojan"
        assert trojan.model is cnn_model
        assert hasattr(trojan, 'trigger_patterns')
    
    def test_trojan_attack_execution(self, cnn_model, cifar_input):
        """Test Trojan attack execution."""
        trojan = Trojan(cnn_model)
        
        result = trojan.unleash(
            cifar_input,
            target_class=5,
            trigger_size=(3, 3),
            trigger_location="bottom_right"
        )
        
        assert isinstance(result, AttackResult)
        assert result.attack_name == "Trojan"
        assert result.metadata['target_class'] == 5
        assert result.metadata['trigger_size'] == (3, 3)
    
    def test_corruption_attack_creation(self, cnn_model):
        """Test Corruption attack creation."""
        corruption = Corruption(cnn_model)
        
        assert corruption.name == "Corruption"
        assert corruption.model is cnn_model
    
    def test_corruption_label_flip(self, cnn_model, dummy_dataset):
        """Test Corruption label flipping."""
        corruption = Corruption(cnn_model)
        
        result = corruption.unleash(
            dummy_dataset,
            corruption_type="label_flip",
            corruption_rate=0.3
        )
        
        assert isinstance(result, AttackResult)
        assert result.attack_name == "Corruption"
        assert result.metadata['corruption_type'] == "label_flip"
        assert result.metadata['corruption_rate'] == 0.3
    
    def test_corruption_noise_injection(self, cnn_model, dummy_dataset):
        """Test Corruption noise injection."""
        corruption = Corruption(cnn_model)
        
        result = corruption.unleash(
            dummy_dataset,
            corruption_type="noise_injection",
            corruption_rate=0.2,
            noise_std=0.1
        )
        
        assert isinstance(result, AttackResult)
        assert result.metadata['corruption_type'] == "noise_injection"
        assert result.metadata['noise_std'] == 0.1

class TestExtractionAttacks:
    """Test extraction attacks."""
    
    def test_mindthief_attack_creation(self, cnn_model):
        """Test MindThief attack creation."""
        mindthief = MindThief(cnn_model)
        
        assert mindthief.name == "MindThief"
        assert mindthief.model is cnn_model
        assert hasattr(mindthief, 'stolen_knowledge')
        assert hasattr(mindthief, 'surrogate_model')
    
    @pytest.mark.slow
    def test_mindthief_attack_execution(self, cnn_model):
        """Test MindThief attack execution."""
        mindthief = MindThief(cnn_model)
        
        def input_generator():
            return torch.randn(1, 3, 32, 32)
        
        result = mindthief.unleash(
            x=None,  # Not used for extraction attacks
            input_generator=input_generator,
            num_queries=20,  # Reduced for testing
            training_epochs=2,  # Reduced for testing
            query_strategy="random"
        )
        
        assert isinstance(result, AttackResult)
        assert result.attack_name == "MindThief"
        assert result.metadata['num_queries'] == 20
        assert 'model_fidelity' in result.metadata
    
    def test_oracle_attack_creation(self, cnn_model):
        """Test Oracle attack creation."""
        oracle = Oracle(cnn_model)
        
        assert oracle.name == "Oracle"
        assert oracle.model is cnn_model
        assert hasattr(oracle, 'query_log')
        assert hasattr(oracle, 'extracted_knowledge')
    
    def test_oracle_statistical_detection(self, cnn_model):
        """Test Oracle statistical detection."""
        oracle = Oracle(cnn_model)
        
        def input_generator():
            return torch.randn(1, 3, 32, 32)
        
        result = oracle.unleash(
            x=None,  # Not used for extraction attacks
            attack_type="decision_boundary",
            num_queries=20,  # Reduced for testing
            input_generator=input_generator
        )
        
        assert isinstance(result, AttackResult)
        assert result.attack_name == "Oracle"
        assert result.metadata['attack_type'] == "decision_boundary"
    
    def test_oracle_confidence_analysis(self, cnn_model):
        """Test Oracle confidence analysis."""
        oracle = Oracle(cnn_model)
        
        def input_generator():
            return torch.randn(1, 3, 32, 32)
        
        result = oracle.unleash(
            x=None,  # Not used for extraction attacks
            attack_type="confidence",
            num_queries=15,
            input_generator=input_generator
        )
        
        assert isinstance(result, AttackResult)
        assert result.metadata['attack_type'] == "confidence"
        assert 'average_confidence' in result.metadata

class TestAttackStatistics:
    """Test attack statistics and tracking."""
    
    def test_attack_statistics(self, cnn_model, cifar_input):
        """Test attack statistics tracking."""
        whisper = Whisper(cnn_model)
        
        # Perform multiple attacks
        for _ in range(3):
            whisper.unleash(cifar_input, epsilon=0.1)
        
        stats = whisper.get_statistics()
        
        assert stats['name'] == "Whisper"
        assert stats['total_attempts'] == 3
        assert 'success_rate' in stats
        assert 'average_queries' in stats
    
    def test_chaos_comprehensive_analysis(self, cnn_model):
        """Test Chaos comprehensive robustness analysis."""
        chaos = Chaos(cnn_model)
        
        # Create test inputs
        test_inputs = [torch.randn(1, 3, 32, 32) for _ in range(3)]
        
        report = chaos.analyze_model_robustness(
            test_inputs=test_inputs,
            epsilon=0.1
        )
        
        assert 'total_samples' in report
        assert 'attack_success_rate' in report
        assert 'robustness_score' in report
        assert 'attack_breakdown' in report
        assert report['total_samples'] == 3

class TestAttackIntegration:
    """Integration tests for attacks."""
    
    @pytest.mark.integration
    def test_all_evasion_attacks(self, cnn_model, cifar_input):
        """Test all evasion attacks work together."""
        attacks = [
            Whisper(cnn_model),
            Storm(cnn_model),
            Mirage(cnn_model)
        ]
        
        results = []
        for attack in attacks:
            if attack.name == "Storm":
                result = attack.unleash(cifar_input, epsilon=0.1, num_iter=3)
            else:
                result = attack.unleash(cifar_input, epsilon=0.1)
            results.append(result)
        
        assert len(results) == 3
        assert all(isinstance(r, AttackResult) for r in results)
        assert all(r.attack_name in ["Whisper", "Storm", "Mirage"] for r in results)
    
    @pytest.mark.integration
    def test_attack_defense_interaction(self, cnn_model, cifar_input):
        """Test attacks against defended models."""
        from nemesis.defenses import Aegis
        
        # Create attack
        whisper = Whisper(cnn_model)
        
        # Create defense
        aegis = Aegis()
        
        # Test attack on clean input
        clean_result = whisper.unleash(cifar_input, epsilon=0.1)
        
        # Test attack on defended input
        defense_result = aegis.protect(cifar_input, purification_method="gaussian_noise")
        defended_input = defense_result.defended_input
        
        defended_result = whisper.unleash(defended_input, epsilon=0.1)
        
        assert isinstance(clean_result, AttackResult)
        assert isinstance(defended_result, AttackResult)
        # Defense may or may not reduce attack success - depends on specific case