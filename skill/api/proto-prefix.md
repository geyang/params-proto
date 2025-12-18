---
title: "@proto.prefix Decorator"
description: Create singleton configuration classes with namespaced CLI arguments
---

# @proto.prefix Decorator

The `@proto.prefix` decorator creates singleton configuration classes with namespaced CLI arguments.

## Basic Usage

```python
from params_proto import proto

@proto.prefix
class Training:
    """Training hyperparameters."""
    lr: float = 0.001  # Learning rate
    batch_size: int = 32  # Batch size
    epochs: int = 100  # Number of epochs

# Access as class attributes (singleton)
print(Training.lr)  # 0.001
Training.lr = 0.01  # Direct modification
```

## With @proto.cli

`@proto.prefix` classes automatically integrate with `@proto.cli`:

```python
@proto.prefix
class Model:
    """Model configuration."""
    name: str = "resnet50"
    dropout: float = 0.5

@proto.prefix
class Training:
    """Training hyperparameters."""
    lr: float = 0.001
    epochs: int = 100

@proto.cli
def main(seed: int = 42):
    """Train with configuration."""
    print(f"Training {Model.name} with lr={Training.lr}")

if __name__ == "__main__":
    main()
```

### CLI Syntax

```bash
# Prefix.param-name syntax
python train.py --model.name vit --training.lr 0.01

# Help shows grouped options
python train.py --help
```

### Help Output

```
usage: train.py [-h] [--seed INT] [OPTIONS]

Train with configuration.

options:
  -h, --help           show this help message and exit
  --seed INT           Random seed (default: 42)

Model options:
  Model configuration.

  --model.name STR     Model name (default: resnet50)
  --model.dropout FLOAT
                       Dropout rate (default: 0.5)

Training options:
  Training hyperparameters.

  --training.lr FLOAT  Learning rate (default: 0.001)
  --training.epochs INT
                       Number of epochs (default: 100)
```

## Singleton Behavior

`@proto.prefix` creates a **singleton** - there's only one instance:

```python
@proto.prefix
class Config:
    value: int = 0

# All access the same singleton
Config.value = 42
print(Config.value)  # 42

# Even "instances" share state
c1 = Config()
c2 = Config()
c1.value = 100
print(c2.value)  # 100
print(Config.value)  # 100
```

## Override Patterns

### 1. Direct Assignment

```python
Training.lr = 0.01
Training.epochs = 200
```

### 2. CLI Arguments

```bash
python train.py --training.lr 0.01 --training.epochs 200
```

### 3. Context Manager

```python
with proto.bind(Training, lr=0.01, epochs=200):
    # Training.lr == 0.01 inside this block
    train()
# Training.lr restored to original after block
```

### 4. Multiple Bindings

```python
with proto.bind(Training, lr=0.01), proto.bind(Model, name="vit"):
    train()
```

## Boolean Flags in Prefix

```python
@proto.prefix
class Config:
    verbose: bool = False  # --config.verbose to enable
    cuda: bool = True  # --no-config.cuda to disable
```

## Naming Convention

- Class name → lowercase prefix in CLI
- `class Training:` → `--training.param`
- `class ModelConfig:` → `--modelconfig.param`

## Multiple Prefix Classes

```python
@proto.prefix
class Data:
    path: str = "./data"
    workers: int = 4

@proto.prefix
class Model:
    name: str = "resnet"
    layers: int = 50

@proto.prefix
class Optimizer:
    name: str = "adam"
    lr: float = 0.001

@proto.cli
def train():
    print(f"Data: {Data.path}")
    print(f"Model: {Model.name}")
    print(f"Optimizer: {Optimizer.name}, lr={Optimizer.lr}")
```

## vs @proto

| Feature | `@proto.prefix` | `@proto` |
|---------|-----------------|----------|
| Singleton | Yes | No |
| CLI integration | Auto with `@proto.cli` | Manual |
| Access pattern | `Class.attr` | `instance.attr` |
| Multiple instances | No | Yes |

Use `@proto.prefix` for:
- Global configuration
- Multi-namespace CLI apps
- Shared state across modules

Use `@proto` for:
- Reusable config templates
- Multiple config instances
- Library code

## Methods in @proto.prefix Classes

`@proto.prefix` classes support all Python method types:

```python
@proto.prefix
class Config:
    lr: float = 0.01

    @staticmethod
    def validate_lr(lr: float) -> bool:
        """Staticmethods work correctly."""
        return 0 < lr < 1.0

    @classmethod
    def get_lr(cls):
        """Classmethods receive the instance (singleton behavior)."""
        return cls.lr

    def summary(self):
        """Instance methods work normally."""
        return f"lr={self.lr}"

obj = Config()
obj.validate_lr(0.01)  # True - staticmethod works
obj.get_lr()           # 0.01 - classmethod works
obj.summary()          # "lr=0.01" - instance method works
```

**Note:** Methods can also be inherited from base classes:

```python
class Base:
    @staticmethod
    def helper(x):
        return x * 2

@proto.prefix
class Config(Base):
    value: int = 10

Config().helper(21)  # 42 - inherited staticmethod works
```

## Post-Initialization Hook

Use `__post_init__` for validation and computed attributes:

```python
@proto.prefix
class Config:
    lr: float = 0.01
    batch_size: int = 32
    total_samples: int = None

    def __post_init__(self):
        # Validation
        if self.lr > 1:
            raise ValueError("lr must be <= 1")
        # Computed attributes
        self.total_samples = self.batch_size * 100

obj = Config()
print(obj.total_samples)  # 3200
```

`__post_init__` runs after all attributes are set, similar to dataclasses.

## Best Practices

1. **Group related params** - One class per logical group
2. **Use docstrings** - Become section descriptions in help
3. **Keep names short** - `Model` not `ModelConfiguration`
4. **Document with comments** - Each param gets help text
