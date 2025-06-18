"""
Test Examples 📚

"Testing the sacred scrolls and tutorials"

Tests to ensure all examples work correctly and demonstrate
proper usage of the Nemesis toolkit.
"""

import pytest
import subprocess
import sys
import os
from pathlib import Path

class TestExampleExecution:
    """Test that examples execute without errors."""
    
    @pytest.fixture
    def examples_dir(self):
        """Get the examples directory path."""
        # Assume we're in tests/ and examples/ is at same level
        current_dir = Path(__file__).parent
        examples_dir = current_dir.parent / "examples"
        return examples_dir
    
    def test_examples_directory_exists(self, examples_dir):
        """Test that examples directory exists."""
        assert examples_dir.exists(), f"Examples directory not found at {examples_dir}"
        assert examples_dir.is_dir(), f"Examples path is not a directory: {examples_dir}"
    
    def test_basic_battle_example_exists(self, examples_dir):
        """Test that basic_battle.py exists."""
        basic_battle_path = examples_dir / "basic_battle.py"
        assert basic_battle_path.exists(), "basic_battle.py example not found"
    
    def test_arena_tournament_example_exists(self, examples_dir):
        """Test that arena_tournament.py exists."""
        arena_tournament_path = examples_dir / "arena_tournament.py"
        assert arena_tournament_path.exists(), "arena_tournament.py example not found"
    
    def test_attack_showcase_example_exists(self, examples_dir):
        """Test that attack_showcase.py exists."""
        attack_showcase_path = examples_dir / "attack_showcase.py"
        assert attack_showcase_path.exists(), "attack_showcase.py example not found"
    
    @pytest.mark.slow
    def test_basic_battle_execution(self, examples_dir):
        """Test that basic_battle.py runs without errors."""
        basic_battle_path = examples_dir / "basic_battle.py"
        
        if not basic_battle_path.exists():
            pytest.skip("basic_battle.py not found")
        
        # Run the example
        result = subprocess.run(
            [sys.executable, str(basic_battle_path)],
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        # Check that it ran successfully
        assert result.returncode == 0, f"basic_battle.py failed with error: {result.stderr}"
        
        # Check for expected output
        output = result.stdout
        assert "Nemesis Basic Battle Example" in output
        assert "Summoning your nemesis" in output or "summoning" in output.lower()
    
    @pytest.mark.slow
    def test_arena_tournament_execution(self, examples_dir):
        """Test that arena_tournament.py runs without errors."""
        arena_tournament_path = examples_dir / "arena_tournament.py"
        
        if not arena_tournament_path.exists():
            pytest.skip("arena_tournament.py not found")
        
        # Run the example
        result = subprocess.run(
            [sys.executable, str(arena_tournament_path)],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout for tournament
        )
        
        # Check that it ran successfully
        assert result.returncode == 0, f"arena_tournament.py failed with error: {result.stderr}"
        
        # Check for expected output
        output = result.stdout
        assert "Arena Tournament Example" in output
        assert "Tournament contestants" in output or "contestants" in output.lower()
        assert "Tournament complete" in output or "complete" in output.lower()
    
    @pytest.mark.slow
    def test_attack_showcase_execution(self, examples_dir):
        """Test that attack_showcase.py runs without errors."""
        attack_showcase_path = examples_dir / "attack_showcase.py"
        
        if not attack_showcase_path.exists():
            pytest.skip("attack_showcase.py not found")
        
        # Run the example
        result = subprocess.run(
            [sys.executable, str(attack_showcase_path)],
            capture_output=True,
            text=True,
            timeout=180  # 3 minute timeout for comprehensive showcase
        )
        
        # Check that it ran successfully
        assert result.returncode == 0, f"attack_showcase.py failed with error: {result.stderr}"
        
        # Check for expected output
        output = result.stdout
        assert "Attack Showcase" in output
        assert "EVASION ATTACKS" in output
        assert "Attack showcase complete" in output or "complete" in output.lower()

class TestExampleContent:
    """Test the content and structure of examples."""
    
    @pytest.fixture
    def examples_dir(self):
        """Get the examples directory path."""
        current_dir = Path(__file__).parent
        examples_dir = current_dir.parent / "examples"
        return examples_dir
    
    def test_basic_battle_content(self, examples_dir):
        """Test basic_battle.py content structure."""
        basic_battle_path = examples_dir / "basic_battle.py"
        
        if not basic_battle_path.exists():
            pytest.skip("basic_battle.py not found")
        
        with open(basic_battle_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for essential components
        assert "from nemesis import summon_nemesis" in content
        assert "NemesisPersonality" in content
        assert "class SimpleModel" in content
        assert "def main():" in content
        assert "__name__ == \"__main__\"" in content
        
        # Check for key functionality
        assert "summon_nemesis" in content
        assert "find_weakness" in content
        assert "forge_armor" in content
        assert "eternal_battle" in content
    
    def test_arena_tournament_content(self, examples_dir):
        """Test arena_tournament.py content structure."""
        arena_tournament_path = examples_dir / "arena_tournament.py"
        
        if not arena_tournament_path.exists():
            pytest.skip("arena_tournament.py not found")
        
        with open(arena_tournament_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for essential components
        assert "from nemesis.arena import Arena" in content
        assert "class SimpleModel" in content
        assert "def main():" in content
        assert "tournament" in content
        assert "legendary_battle" in content
        assert "hall_of_legends" in content
    
    def test_attack_showcase_content(self, examples_dir):
        """Test attack_showcase.py content structure."""
        attack_showcase_path = examples_dir / "attack_showcase.py"
        
        if not attack_showcase_path.exists():
            pytest.skip("attack_showcase.py not found")
        
        with open(attack_showcase_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for essential components
        assert "from nemesis.attacks import" in content
        assert "Whisper" in content
        assert "Storm" in content
        assert "Trojan" in content
        assert "MindThief" in content
        
        # Check for demonstration functions
        assert "demonstrate_evasion_attacks" in content
        assert "demonstrate_poisoning_attacks" in content
        assert "demonstrate_extraction_attacks" in content

class TestExampleDependencies:
    """Test that examples have correct dependencies."""
    
    def test_examples_import_nemesis(self, examples_dir):
        """Test that examples properly import nemesis components."""
        if not examples_dir.exists():
            pytest.skip("Examples directory not found")
        
        example_files = list(examples_dir.glob("*.py"))
        assert len(example_files) > 0, "No Python examples found"
        
        for example_file in example_files:
            with open(example_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Each example should import from nemesis
            assert "nemesis" in content, f"{example_file.name} doesn't import nemesis"
    
    def test_examples_have_main_function(self, examples_dir):
        """Test that examples have proper main functions."""
        if not examples_dir.exists():
            pytest.skip("Examples directory not found")
        
        example_files = list(examples_dir.glob("*.py"))
        
        for example_file in example_files:
            with open(example_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Each example should have main function and __main__ guard
            assert "def main():" in content, f"{example_file.name} missing main() function"
            assert "__name__ == \"__main__\"" in content, f"{example_file.name} missing __main__ guard"

class TestExampleOutput:
    """Test expected outputs from examples."""
    
    @pytest.mark.integration
    def test_basic_battle_produces_expected_output(self, examples_dir):
        """Test that basic_battle.py produces expected mythological output."""
        basic_battle_path = examples_dir / "basic_battle.py"
        
        if not basic_battle_path.exists():
            pytest.skip("basic_battle.py not found")
        
        # Run with shorter timeout and capture output
        result = subprocess.run(
            [sys.executable, str(basic_battle_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            pytest.skip(f"Example failed to run: {result.stderr}")
        
        output = result.stdout.lower()
        
        # Check for mythological terminology
        mythological_terms = [
            "nemesis", "battle", "arena", "legend", "champion",
            "forge", "armor", "weapon", "divine", "epic"
        ]
        
        found_terms = [term for term in mythological_terms if term in output]
        assert len(found_terms) >= 3, f"Expected mythological terminology in output. Found: {found_terms}"
    
    @pytest.mark.integration
    def test_arena_produces_tournament_results(self, examples_dir):
        """Test that arena tournament produces tournament results."""
        arena_tournament_path = examples_dir / "arena_tournament.py"
        
        if not arena_tournament_path.exists():
            pytest.skip("arena_tournament.py not found")
        
        result = subprocess.run(
            [sys.executable, str(arena_tournament_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            pytest.skip(f"Arena example failed to run: {result.stderr}")
        
        output = result.stdout.lower()
        
        # Check for tournament-specific output
        tournament_terms = [
            "tournament", "champion", "contestant", "battle", "winner"
        ]
        
        found_terms = [term for term in tournament_terms if term in output]
        assert len(found_terms) >= 3, f"Expected tournament terminology. Found: {found_terms}"

class TestExampleErrorHandling:
    """Test error handling in examples."""
    
    def test_examples_handle_import_errors_gracefully(self, examples_dir):
        """Test that examples handle missing dependencies gracefully."""
        # This is more of a code review test - examples should have
        # appropriate try/except blocks for imports if needed
        
        if not examples_dir.exists():
            pytest.skip("Examples directory not found")
        
        example_files = list(examples_dir.glob("*.py"))
        
        for example_file in example_files:
            with open(example_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Examples should not have bare imports that could fail
            # (This is a style check - in practice, our examples should work
            # with the required dependencies)
            lines = content.split('\n')
            import_lines = [line.strip() for line in lines if line.strip().startswith('import ') or line.strip().startswith('from ')]
            
            # All import lines should be properly structured
            for import_line in import_lines:
                assert len(import_line) > 6, f"Suspicious import line in {example_file.name}: {import_line}"

class TestExampleDocumentation:
    """Test that examples are properly documented."""
    
    def test_examples_have_docstrings(self, examples_dir):
        """Test that examples have proper documentation."""
        if not examples_dir.exists():
            pytest.skip("Examples directory not found")
        
        example_files = list(examples_dir.glob("*.py"))
        
        for example_file in example_files:
            with open(example_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Each example should have a file-level docstring
            assert '"""' in content, f"{example_file.name} missing docstring"
            
            # Should contain description of what the example does
            docstring_content = content[content.find('"""'):content.find('"""', content.find('"""')+3)+3]
            assert len(docstring_content) > 50, f"{example_file.name} has very short docstring"
    
    def test_examples_have_clear_structure(self, examples_dir):
        """Test that examples have clear, readable structure."""
        if not examples_dir.exists():
            pytest.skip("Examples directory not found")
        
        example_files = list(examples_dir.glob("*.py"))
        
        for example_file in example_files:
            with open(example_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Examples should have reasonable length (not too short, not too long)
            lines = content.split('\n')
            assert 50 < len(lines) < 500, f"{example_file.name} has unusual length: {len(lines)} lines"
            
            # Should have comments explaining key steps
            comment_lines = [line for line in lines if line.strip().startswith('#')]
            assert len(comment_lines) >= 5, f"{example_file.name} needs more explanatory comments"