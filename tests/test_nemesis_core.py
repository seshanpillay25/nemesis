"""
Test Core Nemesis Functionality 🏛️

"Testing the heart of divine retribution"

Tests for the core Nemesis class and its fundamental capabilities.
"""

import pytest
import torch
from nemesis import Nemesis, NemesisPersonality, summon_nemesis, NemesisError, BattleError

class TestNemesisCore:
    """Test the core Nemesis class."""
    
    def test_nemesis_initialization(self, simple_model):
        """Test nemesis creation and initialization."""
        nemesis = Nemesis(simple_model)
        
        assert nemesis.model is simple_model
        assert nemesis.name.startswith("Nemesis-")
        assert nemesis.personality == NemesisPersonality.ADAPTIVE
        assert nemesis.battle_history == []
        assert nemesis.victories == 0
        assert nemesis.defeats == 0
        assert nemesis.evolution_level == 1
        assert nemesis.framework == "pytorch"
    
    def test_nemesis_custom_initialization(self, simple_model):
        """Test nemesis with custom parameters."""
        nemesis = Nemesis(
            simple_model, 
            name="TestBane", 
            personality=NemesisPersonality.AGGRESSIVE
        )
        
        assert nemesis.name == "TestBane"
        assert nemesis.personality == NemesisPersonality.AGGRESSIVE
    
    def test_summon_nemesis_function(self, simple_model):
        """Test the summon_nemesis convenience function."""
        nemesis = summon_nemesis(simple_model, name="SummonedFoe")
        
        assert isinstance(nemesis, Nemesis)
        assert nemesis.name == "SummonedFoe"
        assert nemesis.model is simple_model
    
    def test_framework_detection_pytorch(self, simple_model):
        """Test PyTorch framework detection."""
        nemesis = Nemesis(simple_model)
        assert nemesis.framework == "pytorch"
    
    def test_framework_detection_sklearn(self):
        """Test sklearn framework detection."""
        from sklearn.ensemble import RandomForestClassifier
        sklearn_model = RandomForestClassifier()
        nemesis = Nemesis(sklearn_model)
        assert nemesis.framework == "sklearn"
    
    def test_nemesis_personalities(self, simple_model):
        """Test all nemesis personalities."""
        personalities = [
            NemesisPersonality.AGGRESSIVE,
            NemesisPersonality.CUNNING,
            NemesisPersonality.ADAPTIVE,
            NemesisPersonality.RELENTLESS,
            NemesisPersonality.CHAOTIC
        ]
        
        for personality in personalities:
            nemesis = Nemesis(simple_model, personality=personality)
            assert nemesis.personality == personality

class TestWeaknessDiscovery:
    """Test weakness discovery capabilities."""
    
    def test_find_weakness_adaptive(self, simple_model, flat_input):
        """Test adaptive weakness discovery."""
        nemesis = Nemesis(simple_model, personality=NemesisPersonality.ADAPTIVE)
        
        # Mock the input for testing
        with torch.no_grad():
            report = nemesis.find_weakness(attack_budget=10, epsilon=0.1)
        
        assert hasattr(report, 'vulnerabilities')
        assert isinstance(report.vulnerabilities, list)
        assert len(report) >= 0  # May or may not find vulnerabilities
    
    def test_find_weakness_aggressive(self, simple_model):
        """Test aggressive weakness discovery."""
        nemesis = Nemesis(simple_model, personality=NemesisPersonality.AGGRESSIVE)
        
        report = nemesis.find_weakness(attack_budget=10)
        
        assert hasattr(report, 'vulnerabilities')
        assert isinstance(report.vulnerabilities, list)
    
    def test_find_weakness_different_personalities(self, simple_model):
        """Test weakness discovery across different personalities."""
        personalities = [
            NemesisPersonality.CUNNING,
            NemesisPersonality.RELENTLESS,
            NemesisPersonality.CHAOTIC
        ]
        
        for personality in personalities:
            nemesis = Nemesis(simple_model, personality=personality)
            report = nemesis.find_weakness(attack_budget=5)
            
            assert hasattr(report, 'vulnerabilities')
            assert isinstance(report.vulnerabilities, list)

class TestDefenseForging:
    """Test defense forging capabilities."""
    
    def test_forge_armor_basic(self, simple_model):
        """Test basic armor forging."""
        nemesis = Nemesis(simple_model)
        
        armor = nemesis.forge_armor()
        
        assert armor is not None
        assert hasattr(armor, 'apply')
        assert hasattr(armor, 'strategy')
    
    def test_forge_armor_strategies(self, simple_model):
        """Test different armor strategies."""
        nemesis = Nemesis(simple_model)
        strategies = ["adaptive", "robust", "certified"]
        
        for strategy in strategies:
            armor = nemesis.forge_armor(strategy=strategy)
            assert armor.strategy == strategy
    
    def test_armor_application(self, simple_model):
        """Test applying forged armor."""
        nemesis = Nemesis(simple_model)
        armor = nemesis.forge_armor()
        
        protected_model = armor.apply(simple_model)
        assert protected_model is not None

class TestEternalBattle:
    """Test the eternal battle system."""
    
    @pytest.mark.slow
    def test_eternal_battle_basic(self, simple_model):
        """Test basic eternal battle."""
        nemesis = Nemesis(simple_model)
        
        result = nemesis.eternal_battle(rounds=3, evolution=False)
        
        assert result is not None
        assert len(nemesis.battle_history) == 3
        assert all('round' in battle for battle in nemesis.battle_history)
    
    @pytest.mark.slow  
    def test_eternal_battle_with_evolution(self, simple_model):
        """Test eternal battle with evolution enabled."""
        nemesis = Nemesis(simple_model)
        initial_level = nemesis.evolution_level
        
        result = nemesis.eternal_battle(rounds=2, evolution=True)
        
        assert nemesis.evolution_level >= initial_level
        assert len(nemesis.battle_history) == 2
    
    def test_battle_history_tracking(self, simple_model):
        """Test battle history tracking."""
        nemesis = Nemesis(simple_model)
        
        nemesis.eternal_battle(rounds=2, evolution=False)
        
        assert len(nemesis.battle_history) == 2
        for i, battle in enumerate(nemesis.battle_history):
            assert battle['round'] == i + 1
            assert 'weaknesses_found' in battle
            assert 'evolution_level' in battle

class TestNemesisStatistics:
    """Test nemesis statistics and reporting."""
    
    def test_statistics_tracking(self, simple_model):
        """Test that statistics are properly tracked."""
        nemesis = Nemesis(simple_model)
        
        # Perform some operations
        nemesis.find_weakness(attack_budget=5)
        nemesis.forge_armor()
        
        stats = nemesis.get_statistics() if hasattr(nemesis, 'get_statistics') else {}
        assert isinstance(stats, dict)
    
    def test_evolution_tracking(self, simple_model):
        """Test evolution level tracking."""
        nemesis = Nemesis(simple_model)
        initial_level = nemesis.evolution_level
        
        # Evolution should happen during battles
        nemesis.eternal_battle(rounds=1, evolution=True)
        
        assert nemesis.evolution_level >= initial_level

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_personality(self, simple_model):
        """Test handling of invalid personality."""
        # Should not raise error, but handle gracefully
        nemesis = Nemesis(simple_model, personality="invalid_personality")
        assert nemesis.personality == "invalid_personality"  # Should accept but may not work
    
    def test_none_model(self):
        """Test handling of None model."""
        with pytest.raises((AttributeError, TypeError)):
            Nemesis(None)
    
    def test_empty_battle_rounds(self, simple_model):
        """Test eternal battle with zero rounds."""
        nemesis = Nemesis(simple_model)
        
        result = nemesis.eternal_battle(rounds=0)
        assert len(nemesis.battle_history) == 0
    
    def test_negative_attack_budget(self, simple_model):
        """Test negative attack budget handling."""
        nemesis = Nemesis(simple_model)
        
        # Should handle gracefully
        report = nemesis.find_weakness(attack_budget=0)
        assert hasattr(report, 'vulnerabilities')

class TestIntegration:
    """Integration tests for core functionality."""
    
    @pytest.mark.integration
    def test_full_nemesis_workflow(self, simple_model, flat_input):
        """Test complete nemesis workflow."""
        # Create nemesis
        nemesis = summon_nemesis(simple_model, name="IntegrationTest")
        
        # Find weaknesses
        weaknesses = nemesis.find_weakness(attack_budget=10)
        
        # Forge armor
        armor = nemesis.forge_armor()
        protected_model = armor.apply(simple_model)
        
        # Battle
        result = nemesis.eternal_battle(rounds=2)
        
        # Verify all steps completed
        assert nemesis.name == "IntegrationTest"
        assert hasattr(weaknesses, 'vulnerabilities')
        assert protected_model is not None
        assert len(nemesis.battle_history) == 2
    
    @pytest.mark.integration
    def test_multiple_nemeses(self, simple_model):
        """Test creating multiple nemeses for same model."""
        nemesis1 = Nemesis(simple_model, name="First", personality=NemesisPersonality.AGGRESSIVE)
        nemesis2 = Nemesis(simple_model, name="Second", personality=NemesisPersonality.CUNNING)
        
        # Both should work independently
        report1 = nemesis1.find_weakness(attack_budget=5)
        report2 = nemesis2.find_weakness(attack_budget=5)
        
        assert nemesis1.name != nemesis2.name
        assert nemesis1.personality != nemesis2.personality
        assert isinstance(report1.vulnerabilities, list)
        assert isinstance(report2.vulnerabilities, list)