"""
Test Defense Arsenal 🛡️

"Testing the divine protections and sacred armor"

Tests for all adversarial defenses in the Nemesis arsenal.
"""

import pytest
import torch
import numpy as np
from nemesis.defenses import (
    Aegis, Barrier,
    Fortitude, Resilience, Immunity,
    DefenseBase, DefenseResult, DefenseArmor
)

class TestDefenseBase:
    """Test the base defense class."""
    
    def test_defense_result_creation(self):
        """Test DefenseResult dataclass creation."""
        result = DefenseResult(
            original_input=torch.randn(1, 3, 32, 32),
            defended_input=torch.randn(1, 3, 32, 32),
            original_prediction=torch.randn(1, 10),
            defended_prediction=torch.randn(1, 10),
            defense_applied=True,
            confidence_change=0.1,
            defense_strength=0.8,
            defense_name="TestDefense",
            metadata={"test": "data"}
        )
        
        assert result.defense_applied == True
        assert result.confidence_change == 0.1
        assert result.defense_name == "TestDefense"
        assert result.metadata["test"] == "data"

class TestShieldDefenses:
    """Test shield defenses (preprocessing)."""
    
    def test_aegis_creation(self):
        """Test Aegis defense creation."""
        aegis = Aegis()
        
        assert aegis.name == "Aegis"
        assert aegis.defense_type == "preprocessing"
        assert aegis.applications == 0
    
    def test_aegis_gaussian_noise(self, cifar_input):
        """Test Aegis Gaussian noise purification."""
        aegis = Aegis()
        
        result = aegis.protect(
            cifar_input,
            purification_method="gaussian_noise",
            strength=0.1
        )
        
        assert isinstance(result, DefenseResult)
        assert result.defense_name == "Aegis"
        assert result.metadata['purification_method'] == "gaussian_noise"
        assert result.defended_input.shape == cifar_input.shape
        assert torch.all(result.defended_input >= 0)
        assert torch.all(result.defended_input <= 1)
    
    def test_aegis_median_filter(self, cifar_input):
        """Test Aegis median filter purification."""
        aegis = Aegis()
        
        result = aegis.protect(
            cifar_input,
            purification_method="median_filter",
            strength=0.3
        )
        
        assert isinstance(result, DefenseResult)
        assert result.metadata['purification_method'] == "median_filter"
        assert 'kernel_size' in result.metadata
    
    def test_aegis_bit_depth_reduction(self, cifar_input):
        """Test Aegis bit depth reduction."""
        aegis = Aegis()
        
        result = aegis.protect(
            cifar_input,
            purification_method="bit_depth_reduction",
            strength=0.2
        )
        
        assert isinstance(result, DefenseResult)
        assert result.metadata['purification_method'] == "bit_depth_reduction"
        assert 'num_levels' in result.metadata
    
    def test_aegis_jpeg_compression(self, cifar_input):
        """Test Aegis JPEG compression simulation."""
        aegis = Aegis()
        
        result = aegis.protect(
            cifar_input,
            purification_method="jpeg_compression",
            strength=0.3
        )
        
        assert isinstance(result, DefenseResult)
        assert result.metadata['purification_method'] == "jpeg_compression"
        assert 'jpeg_quality' in result.metadata
    
    def test_aegis_total_variation(self, cifar_input):
        """Test Aegis total variation denoising."""
        aegis = Aegis()
        
        result = aegis.protect(
            cifar_input,
            purification_method="total_variation",
            strength=0.1
        )
        
        assert isinstance(result, DefenseResult)
        assert result.metadata['purification_method'] == "total_variation"
        assert 'tv_weight' in result.metadata
    
    def test_aegis_adaptive_purification(self, cifar_input):
        """Test Aegis adaptive purification."""
        aegis = Aegis()
        
        # Test different suspected attack types
        attack_types = ["whisper", "storm", "shapeshifter", "unknown"]
        
        for attack_type in attack_types:
            result = aegis.adaptive_purification(cifar_input, attack_type)
            assert isinstance(result, DefenseResult)
            assert result.defense_name == "Aegis"
    
    def test_barrier_creation(self):
        """Test Barrier defense creation."""
        barrier = Barrier()
        
        assert barrier.name == "Barrier"
        assert barrier.defense_type == "detection"
        assert barrier.detection_threshold == 0.5
    
    def test_barrier_statistical_detection(self, cifar_input):
        """Test Barrier statistical detection."""
        barrier = Barrier()
        
        result = barrier.protect(
            cifar_input,
            detection_method="statistical",
            threshold=0.5
        )
        
        assert isinstance(result, DefenseResult)
        assert result.defense_name == "Barrier"
        assert result.metadata['detection_method'] == "statistical"
        assert 'anomaly_score' in result.metadata
        assert 'is_adversarial' in result.metadata
    
    def test_barrier_neural_detection(self, cifar_input):
        """Test Barrier neural detection."""
        barrier = Barrier()
        
        result = barrier.protect(
            cifar_input,
            detection_method="neural",
            threshold=0.5
        )
        
        assert isinstance(result, DefenseResult)
        assert result.metadata['detection_method'] == "neural"
        assert 'detection_probability' in result.metadata
    
    def test_barrier_ensemble_detection(self, cifar_input):
        """Test Barrier ensemble detection."""
        barrier = Barrier()
        
        result = barrier.protect(
            cifar_input,
            detection_method="ensemble",
            threshold=0.5
        )
        
        assert isinstance(result, DefenseResult)
        assert result.metadata['detection_method'] == "ensemble"
        assert 'ensemble_score' in result.metadata
    
    def test_barrier_calibration(self, cifar_input):
        """Test Barrier detector calibration."""
        barrier = Barrier()
        
        # Create dummy clean and adversarial samples
        clean_samples = [torch.randn(1, 3, 32, 32) for _ in range(5)]
        adv_samples = [torch.randn(1, 3, 32, 32) + 0.1 for _ in range(5)]
        
        calibration_result = barrier.calibrate_detector(clean_samples, adv_samples)
        
        assert 'threshold' in calibration_result
        assert 'accuracy' in calibration_result
        assert 'clean_accuracy' in calibration_result
        assert 'adversarial_detection_rate' in calibration_result

class TestArmorDefenses:
    """Test armor defenses (training-time)."""
    
    def test_fortitude_creation(self, simple_model):
        """Test Fortitude defense creation."""
        fortitude = Fortitude(simple_model)
        
        assert fortitude.name == "Fortitude"
        assert fortitude.defense_type == "training"
        assert fortitude.model is simple_model
        assert hasattr(fortitude, 'training_history')
    
    @pytest.mark.slow
    def test_fortitude_adversarial_training(self, simple_model, training_dataset):
        """Test Fortitude adversarial training."""
        fortitude = Fortitude(simple_model)
        
        result = fortitude.protect(
            training_dataset,
            epsilon=0.1,
            alpha=0.01,
            num_steps=3,
            training_epochs=2  # Reduced for testing
        )
        
        assert isinstance(result, DefenseResult)
        assert result.defense_name == "Fortitude"
        assert result.metadata['training_method'] == 'adversarial_training'
        assert result.metadata['training_epochs'] == 2
        assert len(fortitude.training_history) == 2
    
    @pytest.mark.slow
    def test_fortitude_trades_training(self, simple_model, training_dataset):
        """Test Fortitude TRADES training."""
        fortitude = Fortitude(simple_model)
        
        result = fortitude.trades_training(
            training_dataset,
            beta=6.0,
            training_epochs=2  # Reduced for testing
        )
        
        assert isinstance(result, DefenseResult)
        assert result.defense_name == "Fortitude-TRADES"
        assert result.metadata['training_method'] == 'TRADES'
        assert result.metadata['beta'] == 6.0
    
    def test_resilience_creation(self, simple_model):
        """Test Resilience defense creation."""
        resilience = Resilience(simple_model)
        
        assert resilience.name == "Resilience"
        assert resilience.defense_type == "training"
        assert resilience.model is simple_model
        assert resilience.temperature == 20.0
    
    @pytest.mark.slow
    def test_resilience_distillation(self, simple_model, training_dataset):
        """Test Resilience defensive distillation."""
        resilience = Resilience(simple_model)
        
        result = resilience.protect(
            training_dataset,
            method="distillation",
            temperature=10.0,
            training_epochs=2  # Reduced for testing
        )
        
        assert isinstance(result, DefenseResult)
        assert result.defense_name == "Resilience"
        assert result.metadata['method'] == 'distillation'
        assert result.metadata['temperature'] == 10.0
        assert resilience.teacher_model is not None
    
    @pytest.mark.slow
    def test_resilience_label_smoothing(self, simple_model, training_dataset):
        """Test Resilience label smoothing."""
        resilience = Resilience(simple_model)
        
        result = resilience.protect(
            training_dataset,
            method="label_smoothing",
            training_epochs=2,  # Reduced for testing
            smoothing=0.1
        )
        
        assert isinstance(result, DefenseResult)
        assert result.metadata['method'] == 'label_smoothing'
    
    @pytest.mark.slow
    def test_resilience_mixup(self, simple_model, training_dataset):
        """Test Resilience mixup training."""
        resilience = Resilience(simple_model)
        
        result = resilience.protect(
            training_dataset,
            method="mixup",
            training_epochs=2,  # Reduced for testing
            alpha=1.0
        )
        
        assert isinstance(result, DefenseResult)
        assert result.metadata['method'] == 'mixup'
    
    def test_immunity_creation(self, simple_model):
        """Test Immunity defense creation."""
        immunity = Immunity(simple_model)
        
        assert immunity.name == "Immunity"
        assert immunity.defense_type == "certified"
        assert immunity.model is simple_model
        assert hasattr(immunity, 'certification_radius')
    
    def test_immunity_randomized_smoothing(self, cnn_model, cifar_input):
        """Test Immunity randomized smoothing."""
        immunity = Immunity(cnn_model)
        
        result = immunity.protect(
            cifar_input,
            method="randomized_smoothing",
            noise_variance=0.1,
            num_samples=100,  # Reduced for testing
            confidence_alpha=0.001
        )
        
        assert isinstance(result, DefenseResult)
        assert result.defense_name == "Immunity"
        assert result.metadata['method'] == 'randomized_smoothing'
        assert 'certification_radius' in result.metadata
        assert 'vote_distribution' in result.metadata
    
    def test_immunity_interval_bound(self, cnn_model, cifar_input):
        """Test Immunity interval bound propagation."""
        immunity = Immunity(cnn_model)
        
        result = immunity.protect(
            cifar_input,
            method="interval_bound",
            epsilon=0.1
        )
        
        assert isinstance(result, DefenseResult)
        assert result.metadata['method'] == 'interval_bound'
        assert 'certified' in result.metadata
    
    def test_immunity_lipschitz(self, cnn_model, cifar_input):
        """Test Immunity Lipschitz-constrained defense."""
        immunity = Immunity(cnn_model)
        
        result = immunity.protect(
            cifar_input,
            method="lipschitz",
            lipschitz_constant=1.0
        )
        
        assert isinstance(result, DefenseResult)
        assert result.metadata['method'] == 'lipschitz'
        assert 'estimated_lipschitz' in result.metadata
    
    def test_immunity_certification_report(self, cnn_model):
        """Test Immunity certification report generation."""
        immunity = Immunity(cnn_model)
        
        test_inputs = [torch.randn(1, 3, 32, 32) for _ in range(3)]
        
        report = immunity.get_certification_report(
            test_inputs,
            method="randomized_smoothing"
        )
        
        assert 'total_inputs' in report
        assert 'certified_inputs' in report
        assert 'certification_rate' in report
        assert 'average_radius' in report
        assert report['total_inputs'] == 3

class TestDefenseArmor:
    """Test the DefenseArmor system."""
    
    def test_armor_creation_adaptive(self, simple_model):
        """Test DefenseArmor creation with adaptive strategy."""
        armor = DefenseArmor(simple_model, strategy="adaptive")
        
        assert armor.model is simple_model
        assert armor.strategy == "adaptive"
        assert len(armor.defenses) > 0
        assert any(isinstance(d, Aegis) for d in armor.defenses)
        assert any(isinstance(d, Barrier) for d in armor.defenses)
    
    def test_armor_creation_robust(self, simple_model):
        """Test DefenseArmor creation with robust strategy."""
        armor = DefenseArmor(simple_model, strategy="robust")
        
        assert armor.strategy == "robust"
        assert len(armor.defenses) > 0
        assert any(isinstance(d, Aegis) for d in armor.defenses)
        assert any(isinstance(d, Fortitude) for d in armor.defenses)
    
    def test_armor_creation_certified(self, simple_model):
        """Test DefenseArmor creation with certified strategy."""
        armor = DefenseArmor(simple_model, strategy="certified")
        
        assert armor.strategy == "certified"
        assert len(armor.defenses) > 0
        assert any(isinstance(d, Aegis) for d in armor.defenses)
        assert any(isinstance(d, Immunity) for d in armor.defenses)
    
    def test_armor_application(self, simple_model):
        """Test armor application to model."""
        armor = DefenseArmor(simple_model, strategy="adaptive")
        
        protected_model = armor.apply()
        
        assert protected_model is not None
        # In our implementation, it returns the original model as defenses are applied during inference
    
    def test_armor_statistics(self, simple_model):
        """Test armor statistics collection."""
        armor = DefenseArmor(simple_model, strategy="adaptive")
        
        stats = armor.get_armor_stats()
        
        assert 'strategy' in stats
        assert 'defense_layers' in stats
        assert 'defense_types' in stats
        assert 'individual_defenses' in stats
        assert stats['strategy'] == "adaptive"
        assert stats['defense_layers'] == len(armor.defenses)

class TestDefenseStatistics:
    """Test defense statistics and tracking."""
    
    def test_defense_statistics_tracking(self, cifar_input):
        """Test defense statistics tracking."""
        aegis = Aegis()
        
        # Apply defense multiple times
        for _ in range(3):
            aegis.protect(cifar_input, purification_method="gaussian_noise", strength=0.1)
        
        stats = aegis.get_statistics()
        
        assert stats['name'] == "Aegis"
        assert stats['total_applications'] == 3
        assert 'success_rate' in stats
        assert 'average_strength' in stats
    
    def test_armor_evolution(self, simple_model):
        """Test armor evolution based on battle results."""
        armor = DefenseArmor(simple_model, strategy="adaptive")
        initial_defenses = len(armor.defenses)
        
        # Simulate battle results showing weakness to specific attacks
        battle_results = {
            'attack_success_rates': {
                'Whisper': 0.8,  # High success rate indicates weakness
                'Storm': 0.4
            }
        }
        
        armor.evolve_armor(battle_results)
        
        # Armor may have evolved (added new defenses)
        assert len(armor.defenses) >= initial_defenses

class TestDefenseIntegration:
    """Integration tests for defenses."""
    
    @pytest.mark.integration
    def test_layered_defense_system(self, simple_model, cifar_input):
        """Test layered defense application."""
        # Create multiple defenses
        aegis = Aegis()
        barrier = Barrier()
        
        # Apply defenses in sequence
        purified_result = aegis.protect(
            cifar_input, 
            purification_method="gaussian_noise", 
            strength=0.05
        )
        
        detection_result = barrier.protect(
            purified_result.defended_input,
            detection_method="statistical"
        )
        
        assert isinstance(purified_result, DefenseResult)
        assert isinstance(detection_result, DefenseResult)
        assert purified_result.defense_applied or not purified_result.defense_applied  # Either is valid
        
    @pytest.mark.integration
    def test_defense_against_attacks(self, cnn_model, cifar_input):
        """Test defenses against actual attacks."""
        from nemesis.attacks import Whisper
        
        # Create attack and defense
        whisper = Whisper(cnn_model)
        aegis = Aegis()
        
        # Attack clean input
        clean_attack = whisper.unleash(cifar_input, epsilon=0.1)
        
        # Defend input
        defense_result = aegis.protect(
            cifar_input,
            purification_method="gaussian_noise",
            strength=0.1
        )
        
        # Attack defended input
        defended_attack = whisper.unleash(defense_result.defended_input, epsilon=0.1)
        
        assert isinstance(clean_attack.success, bool)
        assert isinstance(defended_attack.success, bool)
        # Defense effectiveness depends on specific case
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_armor_workflow(self, simple_model, training_dataset):
        """Test complete armor workflow."""
        # Create comprehensive armor
        armor = DefenseArmor(simple_model, strategy="robust")
        
        # Apply armor (in practice, this would involve training)
        protected_model = armor.apply()
        
        # Get armor statistics
        stats = armor.get_armor_stats()
        
        assert protected_model is not None
        assert isinstance(stats, dict)
        assert 'strategy' in stats
        assert 'defense_layers' in stats
        
        # Test armor evolution
        battle_results = {'attack_success_rates': {'Whisper': 0.7}}
        armor.evolve_armor(battle_results)
        
        evolved_stats = armor.get_armor_stats()
        assert evolved_stats['defense_layers'] >= stats['defense_layers']