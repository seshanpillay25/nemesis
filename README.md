<p align="center">
  <img src="docs/images/nemesis-banner.svg" alt="Nemesis - Your AI's Greatest Adversary" width="800">
</p>

## 🏛️ In Greek mythology, Nemesis was the goddess of retribution

In machine learning, **Nemesis** is your model's personalized adversary - a powerful toolkit that discovers vulnerabilities through battle, making your AI stronger with each confrontation.

Like the ancient goddess who brought divine justice to those who needed humbling, Nemesis challenges your models to grow beyond their limitations, forging unbreakable strength through adversarial fire.

## ⚡ Quick Start

```python
from nemesis import summon_nemesis

# Your model faces its nemesis
nemesis = summon_nemesis(your_model)
weaknesses = nemesis.find_weakness()
armor = nemesis.forge_armor()

# The eternal battle begins
stronger_model = nemesis.eternal_battle()
```

## 🌟 Features

- 🗡️ **20+ Attack Strategies** - From whispers to storms
- 🛡️ **15+ Defense Mechanisms** - Shields and armor for every threat  
- ⚔️ **Automated Battles** - Let your model face its nemesis
- 📊 **Battle Analytics** - Track growth and victories
- 🏆 **Hall of Legends** - Compare your model's journey
- 🏛️ **Arena System** - Epic tournaments and trials
- 🎭 **Nemesis Personalities** - Adaptive, cunning, relentless adversaries
- ⚡ **Evolution System** - Models grow stronger through battle
- 🔮 **Mythological Interface** - Epic theming throughout

## 🏛️ Philosophy

Every model needs a nemesis. Through adversarial confrontation, weaknesses are revealed, defenses are forged, and true robustness emerges. Nemesis doesn't just test your models - it transforms them into legends.

> *"Iron sharpens iron, and one model sharpens another"*

## 🚀 Installation

```bash
git clone https://github.com/seshanpillay25/nemesis.git
cd nemesis
pip install -e .
```

## 🗡️ Attack Arsenal

### Evasion Attacks
- **Whisper** (FGSM) - Subtle perturbations that slip past defenses
- **Storm** (PGD) - Powerful iterative attacks that build like thunder  
- **Shapeshifter** (C&W) - Optimized transformations that bend reality
- **Mirage** (DeepFool) - Minimal perturbations that deceive the eye
- **Chaos** (AutoAttack) - Ensemble mayhem that unleashes multiple strategies

### Poisoning Attacks  
- **Trojan** - Backdoor attacks that hide like the Trojan Horse
- **Corruption** - Label flipping that spreads like plague

### Extraction Attacks
- **MindThief** - Model stealing through clever queries
- **Oracle** - Query-based extraction that reveals hidden knowledge

## 🛡️ Defense Arsenal

### Shields (Preprocessing)
- **Aegis** - Input purification that cleanses corruption
- **Barrier** - Adversarial detection that reveals threats

### Armor (Training-time)
- **Fortitude** - Adversarial training that builds strength through battle
- **Resilience** - Defensive distillation that creates natural resistance  
- **Immunity** - Certified defenses with mathematical guarantees

## ⚔️ Epic Battles

### Face Your Nemesis
```python
from nemesis import Nemesis, NemesisPersonality

# Summon your model's greatest adversary
my_nemesis = Nemesis(model, name="DarkMirror", personality=NemesisPersonality.RELENTLESS)

# Discover vulnerabilities
weaknesses = my_nemesis.find_weakness()
print(f"Nemesis found {len(weaknesses)} weaknesses")

# Forge defenses
armor = my_nemesis.forge_armor(strategy="adaptive")
protected_model = armor.apply(model)

# Eternal battle for ultimate strength
champion = my_nemesis.eternal_battle(rounds=100, evolution=True)
```

### Arena Tournaments
```python
from nemesis.arena import Arena

# Enter the sacred arena
arena = Arena("Colosseum of AI")

# Epic battle between model and nemesis
battle_result = arena.legendary_battle(
    model=your_model,
    rounds=10,
    nemesis_personality="adaptive"
)

# Tournament of champions
tournament_results = arena.tournament([model1, model2, model3])

# View the Hall of Legends
legends = arena.hall_of_legends()
```

### Nemesis Personalities

Each nemesis has a unique fighting style:

```python
# Aggressive - Fast, powerful attacks
nemesis = Nemesis(model, personality=NemesisPersonality.AGGRESSIVE)

# Cunning - Clever, minimal perturbations  
nemesis = Nemesis(model, personality=NemesisPersonality.CUNNING)

# Adaptive - Learns and evolves strategies
nemesis = Nemesis(model, personality=NemesisPersonality.ADAPTIVE)

# Relentless - Never-ending pressure
nemesis = Nemesis(model, personality=NemesisPersonality.RELENTLESS)

# Chaotic - Unpredictable combinations
nemesis = Nemesis(model, personality=NemesisPersonality.CHAOTIC)
```

## 🏛️ Advanced Usage

### Custom Attack Strategies
```python
from nemesis.attacks import Whisper, Storm, Chaos

# Individual attacks
whisper = Whisper(model)
result = whisper.unleash(input_data, epsilon=0.1)

# Ensemble chaos
chaos = Chaos(model)
devastating_result = chaos.unleash(input_data, targeted=True, target_class=5)
```

### Defense Combinations
```python
from nemesis.defenses import Aegis, Fortitude, DefenseArmor

# Layer multiple defenses
armor = DefenseArmor(model, strategy="robust")
fortified_model = armor.apply(model)

# Test protection
protection_report = armor.test_protection(test_inputs, attacks=[whisper, storm])
```

### Battle Analytics
```python
# Detailed weakness analysis
report = nemesis.find_weakness(attack_budget=5000)
for vulnerability in report.vulnerabilities:
    print(f"Attack: {vulnerability['attack_type']}")
    print(f"Success Rate: {vulnerability['success_rate']:.2f}")
    print(f"Severity: {vulnerability['severity']}")

# Model evolution tracking
for round_num, history in enumerate(nemesis.battle_history):
    print(f"Round {round_num}: {history['weaknesses_found']} weaknesses found")
    print(f"Evolution Level: {history['evolution_level']}")
```

## 📊 Robustness Evaluation

```python
# Comprehensive robustness assessment
from nemesis.attacks import Chaos

chaos = Chaos(model)
robustness_report = chaos.analyze_model_robustness(
    test_inputs=test_dataset,
    epsilon=0.1
)

print(f"Attack Success Rate: {robustness_report['attack_success_rate']:.2f}")
print(f"Robustness Score: {robustness_report['robustness_score']:.2f}")
print(f"Average Queries: {robustness_report['average_queries']:.0f}")
```

## 🏆 Model Zoo & Examples

Nemesis comes with pre-trained adversaries for common architectures:

```python
# Load pre-trained nemesis for ResNet
nemesis = summon_nemesis(resnet_model, name="ResNetBane")

# Use community-trained adversaries
from nemesis.zoo import load_nemesis
community_nemesis = load_nemesis("bert-crusher-v2")
```

## 🎭 Visualization & Monitoring

```python
# Battle visualization
nemesis.visualize_battle_scars(input_image)

# Real-time battle monitoring
with nemesis.battle_monitor() as monitor:
    result = nemesis.eternal_battle(rounds=50)
    monitor.save_battle_replay("epic_battle.gif")

# Arena statistics
arena.plot_tournament_results()
arena.export_battle_analytics("arena_stats.csv")
```

## ⚙️ Configuration

Create a `nemesis_config.yaml` for custom setups:

```yaml
nemesis:
  default_personality: "adaptive"
  battle_rounds: 10
  evolution_enabled: true
  
arena:
  name: "Custom Arena"
  auto_save: true
  replay_battles: true

attacks:
  budget: 1000
  epsilon_range: [0.01, 0.3]
  
defenses:
  auto_evolve: true
  protection_level: "high"
```

## 🌐 Framework Support

Nemesis works seamlessly with:

- **PyTorch** - Full support for all features
- **TensorFlow** - Complete compatibility  
- **JAX** - Experimental support
- **Scikit-learn** - Basic adversarial testing
- **Hugging Face** - Transformer model support

## 📚 Research & Citations

Nemesis implements and extends many seminal works in adversarial ML:

### Implemented Methods

- **FGSM** - Goodfellow et al. (2014)
- **PGD** - Madry et al. (2017)  
- **C&W** - Carlini & Wagner (2017)
- **AutoAttack** - Croce & Hein (2020)
- **Randomized Smoothing** - Cohen et al. (2019)
- And many more...

## 🛡️ Security & Ethics

Nemesis is designed for:
- ✅ Improving model robustness
- ✅ Academic research
- ✅ Security testing with permission
- ✅ Educational purposes

Not intended for:
- ❌ Malicious attacks
- ❌ Unauthorized testing
- ❌ Harmful applications

## 🏛️ Acknowledgments

Nemesis stands on the shoulders of giants:

- The adversarial ML research community
- Open source ML frameworks
- Greek mythology for epic inspiration

---

*"In the eternal dance between sword and shield, model and adversary, legends are born. Face your nemesis, embrace the challenge, and emerge victorious."*

**⚔️ May your models be ever stronger ⚔️**