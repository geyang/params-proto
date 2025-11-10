# Configuration Basics

params-proto v3 offers two ways to define configurations: **function-based** and **class-based**. Both use type hints and decorators, but serve different purposes.

## Quick Comparison

| Aspect | Function-Based | Class-Based |
|--------|----------------|-------------|
| **Decorator** | `@proto.cli` | `@proto` or `@proto.prefix` |
| **Best for** | Script entry points, CLI tools | Reusable configs, multiple instances |
| **CLI** | Automatic | Manual (via `@proto.cli` wrapper) |
| **Instances** | One per call | Create as many as needed |
| **Access** | Parameters | Attributes |

**Rule of thumb:** Use functions for scripts, classes for libraries and reusable components.

## Function-Based Configurations

### Basic CLI Function

The simplest way to create a CLI:

```python
from params_proto import proto

@proto.cli
def train(
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Batch size
    epochs: int = 100,  # Number of epochs
):
    """Train a model."""
    print(lr, batch_size, epochs)

if __name__ == "__main__":
    train()
```

**CLI usage:**
```bash
python train.py --lr 0.01 --batch-size 64 --epochs 200
```

### When to Use Functions

✅ **Perfect for:**
- Script entry points (`if __name__ == "__main__"`)
- CLI tools and utilities
- Simple configuration with few parameters
- When you need immediate CLI integration

❌ **Not ideal for:**
- Reusable configuration objects
- When you need multiple instances
- Library code (use `@proto` classes instead)

### Documentation Extraction

Function documentation comes from two sources:

**1. Inline comments:**
```python
@proto.cli
def train(
    lr: float = 0.001,  # This becomes the help text
    batch_size: int = 32,  # Short and sweet
):
    pass
```

**2. Docstring Args section:**
```python
@proto.cli
def train(
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Batch size
):
    """Train a model.

    Args:
        lr: Learning rate for the optimizer. Start with 0.001 and adjust
            based on convergence. Higher values train faster but may be unstable.
        batch_size: Training batch size. Larger values use more memory but
            provide more stable gradients.
    """
    pass
```

The help text combines both: inline comment first, then docstring details.

### Required Parameters

Use Union types with required parameters for subcommand-like behavior:

```python
from dataclasses import dataclass

@dataclass
class Train:
    lr: float = 0.001
    epochs: int = 100

@dataclass
class Evaluate:
    model: str  # Required!
    batch_size: int = 64

@proto.cli
def tool(
    command: Train | Evaluate,  # Required - no default
    verbose: bool = False,
):
    """Multi-command tool."""
    if isinstance(command, Train):
        print(f"Training: lr={command.lr}")
    elif isinstance(command, Evaluate):
        print(f"Evaluating: {command.model}")

if __name__ == "__main__":
    tool()
```

**CLI usage:**
```bash
python tool.py train --lr 0.01
python tool.py evaluate --model checkpoint.pt
```

See [Types Guide](types.md#required-parameters-and-callable-types) for details.

## Class-Based Configurations

### Basic Configuration Class

Create reusable configuration objects:

```python
from params_proto import proto

@proto
class TrainConfig:
    """Training configuration."""
    lr: float = 0.001  # Learning rate
    batch_size: int = 32  # Batch size
    epochs: int = 100  # Number of epochs
```

### Creating and Using Instances

```python
# Create instances with defaults
config1 = TrainConfig()
print(config1.lr)  # 0.001

# Create with custom values
config2 = TrainConfig(lr=0.01, epochs=200)
print(config2.lr)  # 0.01

# Instances are independent
config1.lr = 0.001
config2.lr = 0.01
print(config1.lr, config2.lr)  # 0.001 0.01
```

### Class-Level Access

```python
# Access class defaults
print(TrainConfig.lr)  # 0.001

# Modify class default (affects new instances)
TrainConfig.lr = 0.01

config3 = TrainConfig()
print(config3.lr)  # 0.01 (uses new default)
```

### When to Use Classes

✅ **Perfect for:**
- Reusable configuration objects
- Library code and frameworks
- When you need multiple independent instances
- Configuration hierarchies (inheritance)
- Integration with dataclasses

❌ **Not ideal for:**
- Simple scripts (use `@proto.cli` functions)
- Direct CLI entry points (wrap with `@proto.cli`)

### Inheritance

Build configuration hierarchies:

```python
@proto
class BaseConfig:
    """Base configuration."""
    lr: float = 0.001
    batch_size: int = 32

@proto
class AdamConfig(BaseConfig):
    """Adam optimizer configuration."""
    beta1: float = 0.9
    beta2: float = 0.999

# Inherits lr and batch_size
config = AdamConfig()
print(config.lr, config.beta1)  # 0.001 0.9
```

### Dataclass Integration

Combine with dataclasses for extra features:

```python
from dataclasses import dataclass

@proto
@dataclass
class Params:
    """Model configuration."""
    hidden_size: int = 256
    num_layers: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        """Validate after initialization."""
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be in [0, 1]")

# Dataclass features work
config = Params(hidden_size=512)
print(config)  # Params(hidden_size=512, num_layers=4, dropout=0.1)
```

### Using Classes with CLI

Wrap class instances in a `@proto.cli` function:

```python
@proto
class TrainConfig:
    lr: float = 0.001
    batch_size: int = 32

@proto.cli
def train(config: TrainConfig = TrainConfig()):
    """Train with configuration."""
    print(f"Training with lr={config.lr}")

if __name__ == "__main__":
    train()
```

Or use `@proto.prefix` for global configuration (see [Advanced Patterns](advanced_patterns.md#prefixed-configurations)).

## Choosing Between Functions and Classes

### Use Functions When:

```python
# ✓ Script entry point
@proto.cli
def main(lr: float = 0.001):
    """Train model."""
    pass

if __name__ == "__main__":
    main()
```

### Use Classes When:

```python
# ✓ Reusable configuration
@proto
class ExperimentConfig:
    seed: int = 42
    name: str = "exp"

# Create multiple experiments
exp1 = ExperimentConfig(seed=1, name="baseline")
exp2 = ExperimentConfig(seed=2, name="ablation")
```

### Use Both Together:

```python
# Classes for config structure
@proto
class ModelConfig:
    hidden_size: int = 256
    num_layers: int = 4

@proto
class DataConfig:
    batch_size: int = 32
    num_workers: int = 4

# Function for CLI entry point
@proto.cli
def train(
    model: ModelConfig = ModelConfig(),
    data: DataConfig = DataConfig(),
    epochs: int = 100,
):
    """Train with structured config."""
    print(f"Model: {model.hidden_size} hidden, {model.num_layers} layers")
    print(f"Data: batch_size={data.batch_size}")

if __name__ == "__main__":
    train()
```

## Type Annotations

All parameters must have type annotations:

```python
@proto.cli
def train(
    lr: float = 0.001,  # ✓ Has type annotation
    epochs: int = 100,  # ✓ Has type annotation
    # count = 10,  # ✗ Missing type annotation (won't work)
):
    pass
```

Supported types:
- Basic: `int`, `float`, `str`, `bool`
- Optional: `str | None`, `Optional[int]`
- Collections: `List[int]`, `Tuple[int, int]`
- Literal: `Literal["adam", "sgd"]`
- Enum: `Optimizer.ADAM`
- Path: `pathlib.Path`
- Union: `Train | Evaluate`

See [Type System](types.md) for complete reference.

## Documentation Best Practices

### 1. Use Inline Comments

```python
# ✓ Good: inline comments for quick reference
@proto.cli
def train(
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Training batch size
    epochs: int = 100,  # Number of epochs
):
    pass
```

### 2. Add Docstring for Details

```python
# ✓ Good: docstring for comprehensive documentation
@proto.cli
def train(
    lr: float = 0.001,  # Learning rate
):
    """Train a neural network.

    Args:
        lr: Learning rate for optimizer. Typical values:
            - 0.001 for Adam (default)
            - 0.01-0.1 for SGD
            Start high and reduce if training is unstable.
    """
    pass
```

### 3. Use Type Hints for Constraints

```python
# ✓ Good: Literal types document valid values
from typing import Literal

@proto.cli
def train(
    optimizer: Literal["adam", "sgd", "rmsprop"] = "adam",  # Optimizer type
):
    """Train with specific optimizers only."""
    pass
```

## Common Patterns

### Factory Pattern

```python
@proto
class Params:
    model_type: str = "resnet"
    hidden_size: int = 256

def create_model(config: Params):
    """Create model from config."""
    if config.model_type == "resnet":
        return ResNet(config.hidden_size)
    elif config.model_type == "transformer":
        return Transformer(config.hidden_size)

config = Params(model_type="transformer")
model = create_model(config)
```

### Configuration Registry

```python
@proto
class ResNetConfig:
    num_layers: int = 50

@proto
class TransformerConfig:
    num_heads: int = 8

CONFIG_REGISTRY = {
    "resnet": ResNetConfig,
    "transformer": TransformerConfig,
}

def get_config(name: str):
    """Get config class by name."""
    return CONFIG_REGISTRY[name]()

config = get_config("resnet")
```

## Related

- [Type System](types.md) - Complete type annotation reference
- [CLI Generation](cli_generation.md) - How names convert to CLI arguments
- [Parameter Overrides](overrides.md) - Ways to override values
- [Advanced Patterns](advanced_patterns.md) - Prefixes and subcommands
