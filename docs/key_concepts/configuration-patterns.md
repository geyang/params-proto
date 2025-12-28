# Configuration Patterns

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

---

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

See [Type System](type-system) for details on required parameters.

---

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

Or use `@proto.prefix` for global configuration (see [Advanced Patterns](advanced-patterns)).

### Methods in Configuration Classes

`@proto` classes can include methods just like regular classes. Methods (`classmethod`, `staticmethod`, and instance methods) work as expected:

```python
@proto
class Config:
    lr: float = 0.01
    batch_size: int = 32

    @classmethod
    def from_preset(cls, preset: str = "default"):
        """Create config from a preset."""
        if preset == "large":
            cls.lr = 0.001
            cls.batch_size = 128
        return cls()

    @staticmethod
    def validate_lr(lr: float) -> bool:
        """Validate learning rate."""
        return 0 < lr < 1.0

    def summary(self):
        """Return config summary."""
        return f"lr={self.lr}, batch_size={self.batch_size}"
```

**Usage:**
```python
# Classmethod receives the correct cls
config = Config.from_preset("large")
print(config.lr)  # 0.001

# Staticmethod works as expected
assert Config.validate_lr(0.01) is True

# Instance methods work normally
config = Config()
print(config.summary())  # lr=0.01, batch_size=32
```

### Post-Initialization Hook

`@proto` classes support `__post_init__` (like dataclasses) for validation and computed attributes:

```python
@proto
class TrainConfig:
    lr: float = 0.01
    batch_size: int = 32
    total_samples: int = None  # Computed

    def __post_init__(self):
        # Validation
        if self.lr > 1:
            raise ValueError("lr must be <= 1")

        # Computed attributes
        self.total_samples = self.batch_size * 100
```

**Usage:**
```python
config = TrainConfig(lr=0.5, batch_size=64)
print(config.total_samples)  # 6400

TrainConfig(lr=2.0)  # Raises ValueError
```

---

## Understanding Function Limitations and Solutions

### The Fundamental Problem

Function-based parameter definitions have a limitation: they break the linkage between parameter definitions and their usage. Functions create a new scope for local variables, disconnecting parameters from their original definitions.

Consider:

```python
@proto.cli
def train(lr: float = 0.001, batch_size: int = 32):
    """Train a model."""
    print(f"Learning Rate: {lr}")
    # ❌ No way to link 'lr' back to a centralized parameter definition
```

In vanilla Python, you cannot easily access or iterate over function parameter defaults like you can with class attributes.

### How params-proto Solves This: ProtoWrapper

When you decorate a function with `@proto`, params-proto doesn't just inspect the function—it **wraps** it in a special `ProtoWrapper` object. This wrapper provides the attribute access interface that vanilla Python functions lack.

The `ProtoWrapper` intercepts attribute access and function calls to enable the same ergonomic API that classes provide:

```python
@proto
def train(lr: float = 0.01, batch_size: int = 32):
    print(f"Training with lr={lr}, batch_size={batch_size}")

# ProtoWrapper allows this:
train.lr = 0.001          # Store override
print(train.lr)           # Read current value → 0.001

# And enables sweeps like this:
for train.lr in [0.001, 0.01, 0.1]:
    train()  # Each call uses the updated lr value
```

Behind the scenes:
- **Parameter defaults** are extracted from the function signature and stored internally
- **Overrides** are tracked in a separate dictionary
- **Attribute access** checks overrides first, then falls back to defaults
- **Function calls** merge defaults, overrides, and any kwargs before passing them to the original function

### Alternative Approaches

If you don't want to use `@proto` for functions, here are traditional workarounds:

**Option 1: Argument Data Class**

```python
@dataclass
class TrainParams:
    lr: float = 0.01
    batch_size: int = 32

def train(params: TrainParams) -> None:
    # ✓ Your IDE will link 'params.lr' back to the TrainParams definition
    print(f"Learning Rate: {params.lr}")

config = TrainParams(lr=0.001)
train(params=config)
```

**Option 2: `fn(**kwargs: Unpack)` (Python 3.11+)**

```python
from typing import TypedDict, Unpack

class TrainConfig(TypedDict):
    lr: float
    batch_size: int

def train_with_dict(**kwargs: Unpack[TrainConfig]) -> None:
    print(f"Learning Rate: {kwargs['lr']}")

my_config: TrainConfig = {'lr': 0.001, 'batch_size': 64}
train_with_dict(**my_config)
```

**Option 3: Config Singleton**

```python
@proto
class Config:
    lr: float = 0.01
    batch_size: int = 32

def train() -> None:
    # ✓ Direct access to global configuration
    print(f"Learning Rate: {Config.lr}")

Config.lr = 0.001
train()

# Natural iteration for sweeps:
for Config.lr in [0.01, 0.001, 0.0001]:
    train()
```

---

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

---

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

---

## Related

- [Core Concepts](core-concepts) - The three decorators
- [CLI Fundamentals](cli-fundamentals) - Basic CLI features
- [CLI Patterns](cli-patterns) - Advanced CLI patterns
- [Type System](type-system) - Complete type annotation reference
- [Advanced Patterns](advanced-patterns) - Prefixes and subcommands
- [Parameter Overrides](parameter-overrides) - Ways to override values
- [Hyperparameter Sweeps](hyperparameter-sweeps) - Systematic exploration
