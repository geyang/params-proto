# Core Concepts

params-proto provides three main decorators for different use cases. This guide introduces each one.

## The Three Decorators

### 1. `@proto.cli` - Create a CLI Entry Point

Wraps a function to automatically parse command-line arguments:

```python
from params_proto import proto

@proto.cli
def train(
    learning_rate: float = 0.001,  # Learning rate
    batch_size: int = 32,           # Batch size
):
    """Train a model."""
    print(f"Training with lr={learning_rate}, batch_size={batch_size}")

if __name__ == "__main__":
    train()  # Parses sys.argv and calls train() with parsed args
```

**Usage:**
```bash
python train.py --learning-rate 0.01 --batch-size 64
```

**When to use:** Script entry points, CLI tools, one-off functions.

---

### 2. `@proto` - Define Reusable Configurations

Creates a configuration class that can be instantiated multiple times:

```python
from params_proto import proto
from dataclasses import dataclass

@proto
@dataclass
class Config:
    """Training configuration."""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10

# Create multiple instances with different values
config1 = Config()
config2 = Config(learning_rate=0.01, batch_size=64)
config3 = Config(epochs=50)
```

**When to use:** Libraries, reusable components, multiple instances needed.

---

### 3. `@proto.prefix` - Global Singleton Configurations

Creates singleton configuration groups with automatic CLI prefixes:

```python
from params_proto import proto

@proto.prefix
class Model:
    """Model configuration."""
    name: str = "resnet50"
    hidden_size: int = 256

@proto.prefix
class Training:
    """Training hyperparameters."""
    lr: float = 0.001
    batch_size: int = 32

@proto.cli
def main():
    """Access globally."""
    print(f"Training {Model.name} with lr={Training.lr}")

if __name__ == "__main__":
    main()
```

**CLI usage:**
```bash
python main.py --model.name vit --training.lr 0.01
```

**When to use:** Global configuration namespaces, multiple configuration groups.

---

## Quick Comparison

| Feature | `@proto.cli` | `@proto` | `@proto.prefix` |
|---------|-------------|---------|-----------------|
| **Purpose** | CLI entry point | Reusable config | Global singleton |
| **Scope** | Function wrapper | Class definition | Singleton namespace |
| **Instances** | One per call | Multiple allowed | One global |
| **CLI** | Automatic | Manual (wrap in @proto.cli) | Automatic (with prefix) |
| **Access** | Function params | Instance attributes | Class attributes |
| **Example** | `train()` | `Config()` | `Model.name` |

---

## Common Patterns

### Pattern 1: Simple CLI Function

```python
@proto.cli
def train(lr: float = 0.001, epochs: int = 10):
    """Train model."""
    print(f"Training with lr={lr}, epochs={epochs}")
```

**Best for:** Simple scripts, one-off utilities.

---

### Pattern 2: Reusable Configuration Class

```python
@proto
@dataclass
class TrainConfig:
    lr: float = 0.001
    batch_size: int = 32

@proto.cli
def train(config: TrainConfig):
    """Train with config."""
    print(f"Using config: {config}")
```

**Best for:** Libraries, components used in multiple places.

---

### Pattern 3: Namespaced Global Configs

```python
@proto.prefix
class Model:
    name: str = "resnet50"

@proto.prefix
class Training:
    lr: float = 0.001

@proto.cli
def main():
    """Access as Model.name, Training.lr"""
    pass
```

**Best for:** Large projects with multiple config namespaces.

---

### Pattern 4: Union Types (Subcommands)

```python
@dataclass
class Adam:
    lr: float = 0.001

@dataclass
class SGD:
    lr: float = 0.001
    momentum: float = 0.9

@proto.cli
def train(optimizer: Adam | SGD):
    """Choose optimizer type."""
    if isinstance(optimizer, Adam):
        print(f"Adam with lr={optimizer.lr}")
    else:
        print(f"SGD with lr={optimizer.lr}, momentum={optimizer.momentum}")
```

**CLI usage:**
```bash
python train.py --optimizer:Adam --optimizer.lr 0.01
python train.py sgd --optimizer.momentum 0.95
```

**Best for:** Choosing between different configurations.

---

## Type Annotations

params-proto uses Python's type hints to:
- Generate help text with type information
- Automatically convert CLI string arguments to proper types
- Validate parameter types

**Supported types:**
- Basic: `int`, `float`, `str`, `bool`
- Collections: `List[T]`, `Dict[K, V]`
- Optional: `Optional[T]` (see [Union Types](union-types) for workaround)
- Dataclasses and custom classes
- Enums: `Enum` subclasses

---

## Next Steps

- **For scripts:** Read [Configuration Patterns](configuration-patterns)
- **For advanced use:** Read [Advanced Patterns](advanced-patterns)
- **For CLI details:** Read [CLI Fundamentals](cli-fundamentals)
- **For type details:** Read [Type System](type-system)

---

## Related

- [Welcome](welcome) - Introduction and quick start
- [CLI Fundamentals](cli-fundamentals) - Basic CLI features
- [CLI Patterns](cli-patterns) - Advanced CLI patterns
- [Configuration Patterns](configuration-patterns) - Functions vs classes in depth
- [Union Types](union-types) - Subcommands and optional parameters
- [Advanced Patterns](advanced-patterns) - Prefixes and composition
