---
title: Quick Reference
description: Cheat sheet for params-proto patterns and syntax
---

# params-proto Quick Reference

## Installation

```bash
pip install params-proto==3.0.0-rc23
```

## Decorators

```python
from params_proto import proto

# CLI entry point - parses sys.argv
@proto.cli
def main(lr: float = 0.001): ...

# Singleton config - access as ClassName.attr
@proto.prefix
class Config:
    lr: float = 0.001

# Multi-instance config
@proto
class Params:
    lr: float = 0.001
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
| `Optional[T]` | `VALUE` | `path: str \| None = None` |
| `Path` | `VALUE` | `dir: Path = Path(".")` |

## Boolean Flags

```python
@proto.cli
def train(
    verbose: bool = False,  # --verbose BOOL sets True
    cuda: bool = True,      # --no-cuda BOOL sets False
): ...
```

- Default `False` → help shows `--flag`
- Default `True` → help shows `--no-flag`

## CLI Argument Syntax

```bash
# Named arguments (underscore → hyphen)
python train.py --learning-rate 0.01 --batch-size 64

# Positional for required params
python train.py 42  # First required param

# Boolean flags
python train.py --verbose      # Set to True
python train.py --no-cuda      # Set to False

# Prefix syntax
python train.py --model.name resnet --training.lr 0.01
```

## Environment Variables

```python
from params_proto import proto, EnvVar

@proto.cli
def train(
    lr: float = EnvVar @ "LEARNING_RATE" | 0.001,  # Env var with default
    api_key: str = EnvVar @ "API_KEY",  # Required env var
    host: str = EnvVar @ "HOST" | "localhost",
): ...
```

```bash
LEARNING_RATE=0.01 python train.py
```

## Override Patterns

```python
# 1. CLI
python train.py --lr 0.01

# 2. Direct assignment (for @proto.prefix classes)
Training.lr = 0.01

# 3. Context manager
with proto.bind(Training, lr=0.01):
    train()

# 4. Instance creation
config = Params(lr=0.01)
```

## Help Text Generation

```python
@proto.cli
def train(
    lr: float = 0.001,  # This comment → help text
):
    """This docstring → CLI description."""
```

Output:
```
usage: train.py [-h] [--lr FLOAT]

This docstring → CLI description.

options:
  --lr FLOAT    This comment → help text (default: 0.001)
```

## Multi-Namespace Config

```python
@proto.prefix
class Model:
    """Model configuration."""
    name: str = "resnet50"
    layers: int = 50

@proto.prefix
class Training:
    """Training hyperparameters."""
    lr: float = 0.001
    epochs: int = 100

@proto.cli
def main(seed: int = 42):
    print(f"{Model.name} with lr={Training.lr}")

if __name__ == "__main__":
    main()
```

```bash
python train.py --model.name vit --training.lr 0.0001
```

## Hyperparameter Sweeps

```python
from params_proto import proto, Sweep

@proto.cli
def train(lr: float = 0.001, batch_size: int = 32): ...

# Grid sweep
sweep = Sweep(train).product(
    lr=[0.001, 0.01, 0.1],
    batch_size=[32, 64, 128],
)

for config in sweep:
    train(**config)
```

## Common Patterns

### Required Parameters

```python
@proto.cli
def train(
    seed: int,  # No default = required, shows (required) in help
    lr: float = 0.001,
): ...
```

### Union Types (Subcommand-like)

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
```

```bash
python train.py adam --lr 0.001
python train.py sgd --momentum 0.95
```

### Inheritance

```python
class BaseConfig:
    lr: float = 0.001
    batch_size: int = 32

@proto
class TrainConfig(BaseConfig):
    epochs: int = 100

c = TrainConfig()
vars(c)  # {'lr': 0.001, 'batch_size': 32, 'epochs': 100}
```

Parent fields are included in `vars()` and CLI args.

### Inheritance with EnvVar

EnvVar fields are inherited and type-converted correctly:

```python
class BaseConfig:
    host: str = EnvVar @ "HOST" | "localhost"
    port: int = EnvVar @ "PORT" | 8080

@proto.prefix
class AppConfig(BaseConfig):
    debug: bool = EnvVar @ "DEBUG" | False
```

```bash
HOST=10.0.0.1 PORT=3000 DEBUG=true python app.py
# AppConfig.host = "10.0.0.1" (str)
# AppConfig.port = 3000 (int)
# AppConfig.debug = True (bool)
```

### Post-Init Hook

```python
@proto
class Config:
    lr: float = 0.01
    total: int = None  # Computed

    def __post_init__(self):
        if self.lr > 1:
            raise ValueError("lr too high")
        self.total = int(self.lr * 1000)

c = Config(lr=0.5)
print(c.total)  # 500
```
