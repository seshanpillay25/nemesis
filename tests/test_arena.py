"""
Test Arena System 🏛️

"Testing the sacred battlegrounds where legends are born"

Tests for the Arena system, tournaments, and Hall of Legends.
"""

import pytest
import torch
import torch.nn as nn
from nemesis.arena import Arena, BattleResult
from nemesis import NemesisPersonality

class TestArenaCreation:
    """Test Arena creation and initialization."""
    
    def test_arena_creation(self):
        """Test basic arena creation."""
        arena = Arena("Test Arena")
        
        assert arena.name == "Test Arena"
        assert arena.battle_history == []
        assert arena.active_battles == {}
        assert arena.champion_models == {}
        assert arena.legendary_artifacts == []
    
    def test_arena_default_name(self):
        """Test arena with default name."""
        arena = Arena()
        
        assert arena.name == "Arena of Legends"

class TestNemesisSummoning:
    """Test nemesis summoning in arena."""
    
    def test_summon_nemesis_basic(self, simple_model):
        """Test basic nemesis summoning."""
        arena = Arena()
        
        nemesis = arena.summon_nemesis(simple_model)
        
        assert nemesis.model is simple_model
        assert nemesis.name.startswith("Nemesis-")
        assert nemesis.personality == NemesisPersonality.ADAPTIVE
    
    def test_summon_nemesis_with_name(self, simple_model):
        """Test nemesis summoning with custom name."""
        arena = Arena()
        
        nemesis = arena.summon_nemesis(simple_model, name="ArenaFoe")
        
        assert nemesis.name == "ArenaFoe"
    
    def test_summon_nemesis_with_personality(self, simple_model):
        """Test nemesis summoning with custom personality."""
        arena = Arena()
        
        nemesis = arena.summon_nemesis(
            simple_model, 
            personality=NemesisPersonality.AGGRESSIVE
        )
        
        assert nemesis.personality == NemesisPersonality.AGGRESSIVE

class TestLegendaryBattles:
    """Test legendary battle system."""
    
    def test_legendary_battle_basic(self, simple_model):
        """Test basic legendary battle."""
        arena = Arena()
        
        battle_result = arena.legendary_battle(
            model=simple_model,
            rounds=2,  # Reduced for testing
            evolution_enabled=False
        )
        
        assert isinstance(battle_result, BattleResult)
        assert battle_result.rounds_fought == 2
        assert battle_result.model_name.startswith("Model-") or hasattr(simple_model, 'name')
        assert battle_result.nemesis_name.startswith("Nemesis-")
        assert battle_result.model_victories >= 0
        assert battle_result.nemesis_victories >= 0
        assert battle_result.draws >= 0
        assert 0.0 <= battle_result.final_robustness_score <= 1.0
        assert battle_result.improvement_gained >= 0.0
        assert battle_result.battle_duration > 0.0
        assert isinstance(battle_result.legendary_moments, list)
        assert len(battle_result.battle_id) == 8
    
    def test_legendary_battle_with_evolution(self, simple_model):
        """Test legendary battle with evolution enabled."""
        arena = Arena()
        
        battle_result = arena.legendary_battle(
            model=simple_model,
            rounds=2,
            evolution_enabled=True
        )
        
        assert isinstance(battle_result, BattleResult)
        assert battle_result.rounds_fought == 2
        # Evolution may or may not occur depending on battle outcome
    
    def test_legendary_battle_personalities(self, simple_model):
        """Test legendary battles with different nemesis personalities."""
        arena = Arena()
        personalities = [
            NemesisPersonality.AGGRESSIVE,
            NemesisPersonality.CUNNING,
            NemesisPersonality.RELENTLESS
        ]
        
        for personality in personalities:
            battle_result = arena.legendary_battle(
                model=simple_model,
                rounds=1,
                nemesis_personality=personality
            )
            
            assert isinstance(battle_result, BattleResult)
            assert battle_result.rounds_fought == 1
    
    def test_legendary_battle_custom_name(self, simple_model):
        """Test legendary battle with custom battle name."""
        arena = Arena()
        
        battle_result = arena.legendary_battle(
            model=simple_model,
            rounds=1,
            battle_name="Epic Test Battle"
        )
        
        assert isinstance(battle_result, BattleResult)
        # Battle name is used for display but not stored in result
    
    def test_battle_history_tracking(self, simple_model):
        """Test that battles are tracked in arena history."""
        arena = Arena()
        
        # Conduct multiple battles
        battle1 = arena.legendary_battle(simple_model, rounds=1)
        battle2 = arena.legendary_battle(simple_model, rounds=1)
        
        assert len(arena.battle_history) == 2
        assert arena.battle_history[0] is battle1
        assert arena.battle_history[1] is battle2

class TestTournaments:
    """Test tournament system."""
    
    def test_tournament_basic(self, simple_model, cnn_model):
        """Test basic tournament functionality."""
        arena = Arena()
        
        # Create models with names for better testing
        simple_model.name = "SimpleChampion"
        cnn_model.name = "CNNWarrior"
        
        models = [simple_model, cnn_model]
        
        tournament_results = arena.tournament(
            models=models,
            tournament_name="Test Tournament",
            rounds_per_battle=2  # Reduced for testing
        )
        
        assert isinstance(tournament_results, dict)
        assert tournament_results['tournament_name'] == "Test Tournament"
        assert tournament_results['contestants'] == 2
        assert len(tournament_results['battle_results']) == 2
        assert len(tournament_results['leaderboard']) == 2
        assert tournament_results['champion'] is not None
        
        # Check leaderboard structure
        for entry in tournament_results['leaderboard']:
            assert 'model_name' in entry
            assert 'robustness_score' in entry
            assert 'improvement' in entry
            assert 'victories' in entry
            assert 'legendary_moments' in entry
        
        # Check champion
        champion = tournament_results['champion']
        assert 'model_name' in champion
        assert 'robustness_score' in champion
    
    def test_tournament_single_model(self, simple_model):
        """Test tournament with single model."""
        arena = Arena()
        simple_model.name = "SoloFighter"
        
        tournament_results = arena.tournament(
            models=[simple_model],
            rounds_per_battle=1
        )
        
        assert tournament_results['contestants'] == 1
        assert len(tournament_results['battle_results']) == 1
        assert tournament_results['champion']['model_name'] == "SoloFighter"
    
    def test_tournament_empty_models(self):
        """Test tournament with no models."""
        arena = Arena()
        
        tournament_results = arena.tournament(
            models=[],
            tournament_name="Empty Tournament"
        )
        
        assert tournament_results['contestants'] == 0
        assert len(tournament_results['battle_results']) == 0
        assert len(tournament_results['leaderboard']) == 0
        assert tournament_results['champion'] is None

class TestHallOfLegends:
    """Test Hall of Legends functionality."""
    
    def test_hall_of_legends_empty(self):
        """Test Hall of Legends with no battles."""
        arena = Arena()
        
        legends = arena.hall_of_legends()
        
        assert isinstance(legends, dict)
        assert 'message' in legends
        assert legends['message'] == 'No battles recorded yet'
    
    def test_hall_of_legends_with_battles(self, simple_model):
        """Test Hall of Legends with battle history."""
        arena = Arena()
        simple_model.name = "LegendaryModel"
        
        # Conduct some battles
        arena.legendary_battle(simple_model, rounds=2)
        arena.legendary_battle(simple_model, rounds=1)
        
        legends = arena.hall_of_legends()
        
        assert isinstance(legends, dict)
        assert 'total_battles' in legends
        assert 'greatest_champions' in legends
        assert 'most_legendary_moments' in legends
        assert 'epic_statistics' in legends
        
        assert legends['total_battles'] == 2
        
        # Check epic statistics
        stats = legends['epic_statistics']
        assert 'total_rounds_fought' in stats
        assert 'total_model_victories' in stats
        assert 'total_battle_time' in stats
        assert 'average_battle_duration' in stats
        assert 'overall_model_win_rate' in stats
        
        assert stats['total_rounds_fought'] == 3  # 2 + 1 rounds
    
    def test_hall_of_legends_champions(self, simple_model, cnn_model):
        """Test Hall of Legends champion tracking."""
        arena = Arena()
        simple_model.name = "Champion1"
        cnn_model.name = "Champion2"
        
        # Conduct battles for different models
        arena.legendary_battle(simple_model, rounds=1)
        arena.legendary_battle(cnn_model, rounds=1)
        arena.legendary_battle(simple_model, rounds=1)  # Second battle for Champion1
        
        legends = arena.hall_of_legends()
        
        champions = legends['greatest_champions']
        assert len(champions) <= 5  # Top 5 limit
        
        # Find Champion1 in results
        champion1_stats = next(
            (c for c in champions if c['name'] == 'Champion1'), 
            None
        )
        assert champion1_stats is not None
        assert champion1_stats['battles'] == 2
        assert 'avg_robustness' in champion1_stats
        assert 'avg_improvement' in champion1_stats

class TestArenaUtilities:
    """Test arena utility functions."""
    
    def test_save_arena_state(self, simple_model, temp_dir):
        """Test saving arena state to file."""
        arena = Arena("Saveable Arena")
        
        # Conduct a battle to have some state
        arena.legendary_battle(simple_model, rounds=1)
        
        save_path = temp_dir / "arena_state.json"
        arena.save_arena_state(str(save_path))
        
        assert save_path.exists()
        
        # Check file contents
        import json
        with open(save_path, 'r') as f:
            data = json.load(f)
        
        assert data['name'] == "Saveable Arena"
        assert data['total_battles'] == 1
        assert 'saved_at' in data
        assert 'battle_summary' in data
        assert len(data['battle_summary']) == 1
    
    def test_get_battle_by_id(self, simple_model):
        """Test retrieving battle by ID."""
        arena = Arena()
        
        battle_result = arena.legendary_battle(simple_model, rounds=1)
        battle_id = battle_result.battle_id
        
        retrieved_battle = arena.get_battle_by_id(battle_id)
        
        assert retrieved_battle is battle_result
        assert retrieved_battle.battle_id == battle_id
    
    def test_get_battle_by_invalid_id(self, simple_model):
        """Test retrieving battle with invalid ID."""
        arena = Arena()
        
        arena.legendary_battle(simple_model, rounds=1)
        
        retrieved_battle = arena.get_battle_by_id("invalid_id")
        
        assert retrieved_battle is None

class TestBattleResult:
    """Test BattleResult dataclass."""
    
    def test_battle_result_creation(self):
        """Test BattleResult creation."""
        result = BattleResult(
            model_name="TestModel",
            nemesis_name="TestNemesis",
            rounds_fought=5,
            model_victories=3,
            nemesis_victories=2,
            draws=0,
            final_robustness_score=0.75,
            improvement_gained=0.25,
            battle_duration=10.5,
            legendary_moments=[],
            battle_id="test123"
        )
        
        assert result.model_name == "TestModel"
        assert result.nemesis_name == "TestNemesis"
        assert result.rounds_fought == 5
        assert result.model_victories == 3
        assert result.nemesis_victories == 2
        assert result.draws == 0
        assert result.final_robustness_score == 0.75
        assert result.improvement_gained == 0.25
        assert result.battle_duration == 10.5
        assert result.legendary_moments == []
        assert result.battle_id == "test123"

class TestArenaIntegration:
    """Integration tests for arena system."""
    
    @pytest.mark.integration
    def test_complete_arena_workflow(self, simple_model, cnn_model):
        """Test complete arena workflow."""
        arena = Arena("Integration Arena")
        
        # Set model names
        simple_model.name = "IntegrationModel1"
        cnn_model.name = "IntegrationModel2"
        
        # Individual battles
        battle1 = arena.legendary_battle(simple_model, rounds=1)
        battle2 = arena.legendary_battle(cnn_model, rounds=1)
        
        # Tournament
        tournament_results = arena.tournament([simple_model, cnn_model], rounds_per_battle=1)
        
        # Hall of Legends
        legends = arena.hall_of_legends()
        
        # Verify everything worked
        assert len(arena.battle_history) == 4  # 2 individual + 2 tournament
        assert isinstance(battle1, BattleResult)
        assert isinstance(battle2, BattleResult)
        assert tournament_results['contestants'] == 2
        assert legends['total_battles'] == 4
        assert len(legends['greatest_champions']) >= 1
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_arena_with_nemesis_integration(self, simple_model):
        """Test arena integration with full nemesis functionality."""
        from nemesis import Nemesis
        
        arena = Arena("Nemesis Integration Arena")
        
        # Create nemesis outside arena
        nemesis = Nemesis(simple_model, name="ExternalNemesis")
        
        # Use arena for battle (arena creates its own nemesis)
        battle_result = arena.legendary_battle(
            simple_model,
            rounds=2,
            nemesis_personality=NemesisPersonality.CUNNING
        )
        
        # Verify integration works
        assert isinstance(battle_result, BattleResult)
        assert battle_result.rounds_fought == 2
        assert len(arena.battle_history) == 1
    
    @pytest.mark.integration
    def test_multiple_tournaments(self, simple_model, cnn_model):
        """Test multiple tournaments in same arena."""
        arena = Arena("Multi-Tournament Arena")
        
        # Set names
        simple_model.name = "Fighter1"
        cnn_model.name = "Fighter2"
        
        # First tournament
        tournament1 = arena.tournament([simple_model], tournament_name="Solo Championship")
        
        # Second tournament
        tournament2 = arena.tournament([simple_model, cnn_model], tournament_name="Duo Battle")
        
        # Verify both tournaments
        assert tournament1['tournament_name'] == "Solo Championship"
        assert tournament2['tournament_name'] == "Duo Battle"
        assert len(arena.battle_history) == 3  # 1 + 2 battles
        
        # Hall of Legends should reflect all battles
        legends = arena.hall_of_legends()
        assert legends['total_battles'] == 3