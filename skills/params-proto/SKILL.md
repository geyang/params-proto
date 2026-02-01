---
name: params-proto
description: |
  Declarative hyperparameter management for ML/AI experiments. Use when Claude needs to:
  (1) Create CLI applications with type-hinted parameters and auto-generated help
  (2) Configure ML training scripts with @proto.cli, @proto.prefix, or @proto decorators
  (3) Set up multi-namespace configurations with namespaced CLI arguments
  (4) Read configuration from environment variables using EnvVar
  (5) Create hyperparameter sweeps using piter or Sweep
  (6) Work with Union types for subcommand-like CLI patterns
---

# params-proto v3.2.0

Declarative hyperparameter management for ML experiments with automatic CLI generation.

## Installation

```bash
pip install params-proto==3.2.0
```

## Three Decorators

| Decorator | Purpose | Access Pattern |
|-----------|---------|----------------|
| `@proto.cli` | CLI entry point | Parses sys.argv automatically |
| `@proto.prefix` | Singleton config | `ClassName.attr` (class-level) |
| `@proto` | Multi-instance | `instance.attr` (object-level) |

## Quick Start

### Simple CLI Script

```python
from params_proto import proto

@proto.cli
def train(
    lr: float = 0.001,  # Learning rate (inline comment = help text)
    batch_size: int = 32,  # Batch size
    epochs: int = 100,  # Number of epochs
):
    """Train a model."""  # Docstring = CLI description
    print(f"Training with lr={lr}")

if __name__ == "__main__":
    train()
```

```bash
python train.py --lr 0.01 --batch-size 64
python train.py --help
```

### Multi-Namespace Configuration

```python
@proto.prefix
class Model:
    name: str = "resnet50"  # Architecture
    dropout: float = 0.5  # Dropout rate

@proto.prefix
class Training:
    lr: float = 0.001  # Learning rate
    epochs: int = 100  # Epochs

@proto.cli
def main(seed: int = 42):
    """Train with namespaced config."""
    print(f"Model: {Model.name}, LR: {Training.lr}")

# CLI: python train.py --model.name vit --training.lr 0.01
```

### Environment Variables

```python
from params_proto import proto, EnvVar

@proto.cli
def train(
    lr: float = EnvVar @ "LEARNING_RATE" | 0.001,  # Env var with default
    api_key: str = EnvVar @ "API_KEY",  # Required env var (no default)
    # OR operation: try multiple env vars in order
    token: str = EnvVar @ "API_TOKEN" @ "AUTH_TOKEN" | "default",
): ...
```

### Union Types (Subcommand Pattern)

```python
from dataclasses import dataclass

@dataclass
class Adam:
    lr: float = 0.001
    beta1: float = 0.9

@dataclass
class SGD:
    lr: float = 0.01
    momentum: float = 0.9

@proto.cli
def train(optimizer: Adam | SGD):
    """Train with selected optimizer."""
    print(f"Using {type(optimizer).__name__}")

# CLI: python train.py adam --lr 0.001
# CLI: python train.py sgd --momentum 0.95
```

### Hyperparameter Sweeps with piter

```python
from params_proto.hyper import piter

# Zip (default): pairs values element-wise
configs = piter @ {"lr": [0.001, 0.01], "batch_size": [32, 64]}
# 2 configs: (0.001, 32), (0.01, 64)

# Cartesian product with * (only first needs piter @)
configs = piter @ {"lr": [0.001, 0.01]} * {"batch_size": [32, 64]}
# 4 configs: all combinations

# Override with fixed values using %
configs = piter @ {"lr": [0.001, 0.01]} * {"batch_size": [32, 64]} % {"seed": 42}

# Repeat for multiple trials using **
configs = (piter @ {"lr": [0.001, 0.01]}) ** 3  # 2 configs x 3 trials

for config in configs:
    train(**config)
```

## Type Annotations

| Type | CLI Display | Example |
|------|-------------|---------|
| `int` | `INT` | `count: int = 10` |
| `float` | `FLOAT` | `lr: float = 0.001` |
| `str` | `STR` | `name: str = "default"` |
| `bool` | `BOOL` | `debug: bool = False` |
| `Enum` | `{A,B,C}` | `opt: Optimizer = Optimizer.ADAM` |
| `Literal` | `VALUE` | `mode: Literal["a", "b"] = "a"` |
| `List[T]` | `VALUE` | `ids: List[int] = [1, 2]` |
| `Tuple[T, ...]` | `VALUE` | `dims: Tuple[int, ...] = (224, 224)` |
| `Optional[T]` | `VALUE` | `path: str \| None = None` |

## Boolean Flags

```python
@proto.cli
def train(
    verbose: bool = False,  # --verbose sets True
    cuda: bool = True,      # --no-cuda sets False
): ...
```

## Override Priority (highest to lowest)

1. CLI arguments
2. Direct assignment (`Config.lr = 0.01`)
3. Context manager (`with proto.bind(Config, lr=0.01): ...`)
4. Environment variables
5. Default values

## Getting a Clean Dict

```python
Config._dict      # → {'lr': 0.001, 'batch_size': 32}
dict(Config)      # → same (works for classes and functions)
```

## Reference Files

For detailed documentation, see:

- [cli-and-types.md](https://raw.githubusercontent.com/geyang/params-proto/main/skills/params-proto/references/cli-and-types.md) - @proto.cli, @proto.prefix, type system
- [environment-vars.md](https://raw.githubusercontent.com/geyang/params-proto/main/skills/params-proto/references/environment-vars.md) - EnvVar with templates and inheritance
- [sweeps.md](https://raw.githubusercontent.com/geyang/params-proto/main/skills/params-proto/references/sweeps.md) - piter and Sweep for hyperparameter search
- [patterns.md](https://raw.githubusercontent.com/geyang/params-proto/main/skills/params-proto/references/patterns.md) - Common ML patterns and examples
