# CLI and Types Reference

## Table of Contents

- [@proto.cli Decorator](#protocli-decorator)
- [@proto.prefix Decorator](#protoprefix-decorator)
- [@proto Decorator](#proto-decorator)
- [Type Annotations](#type-annotations)
- [Help Text Generation](#help-text-generation)

---

## @proto.cli Decorator

Creates CLI entry points from functions or classes.

### Basic Usage

```python
from params_proto import proto

@proto.cli
def train(
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Batch size
):
    """Train a neural network."""
    print(f"Training with lr={lr}")

if __name__ == "__main__":
    train()
```

### Parameters

- `prog` - Override program name in help: `@proto.cli(prog="my-trainer")`

### CLI Argument Parsing

```bash
# Named arguments (underscore → hyphen)
python train.py --learning-rate 0.01 --batch-size 64

# Positional for required params (no default)
python train.py 42  # First required param

# Boolean flags
python train.py --verbose      # Set to True
python train.py --no-cuda      # Set to False

# Prefix syntax for @proto.prefix classes
python train.py --model.name resnet --training.lr 0.01
```

### Required vs Optional Parameters

```python
@proto.cli
def train(
    seed: int,  # Required - no default, shows (required) in help
    lr: float = 0.001,  # Optional - has default
): ...
```

### Boolean Flags

```python
@proto.cli
def train(
    verbose: bool = False,  # --verbose sets True
    cuda: bool = True,      # --no-cuda sets False
): ...
```

### With @classmethod and @staticmethod

Place `@proto.cli` on the OUTSIDE (applied last):

```python
class Trainer:
    @proto.cli          # OUTSIDE - receives the descriptor
    @staticmethod
    def evaluate(model_path: str, threshold: float = 0.5): ...

    @proto.cli          # OUTSIDE
    @classmethod
    def train(cls, lr: float = 0.01): ...
```

---

## @proto.prefix Decorator

Creates singleton configuration classes with namespaced CLI arguments.

### Basic Usage

```python
@proto.prefix
class Training:
    """Training hyperparameters."""
    lr: float = 0.001  # Learning rate
    batch_size: int = 32  # Batch size

# Access as class attributes (singleton)
print(Training.lr)  # 0.001
Training.lr = 0.01  # Direct modification
```

### With @proto.cli

```python
@proto.prefix
class Model:
    name: str = "resnet50"
    dropout: float = 0.5

@proto.prefix
class Training:
    lr: float = 0.001
    epochs: int = 100

@proto.cli
def main(seed: int = 42):
    print(f"Training {Model.name} with lr={Training.lr}")

if __name__ == "__main__":
    main()
```

```bash
python train.py --model.name vit --training.lr 0.01
```

### Singleton Behavior

```python
@proto.prefix
class Config:
    value: int = 0

Config.value = 42
c1 = Config()
c2 = Config()
c1.value = 100
print(c2.value)  # 100 - same singleton
```

### Override Patterns

```python
# 1. Direct assignment
Training.lr = 0.01

# 2. CLI arguments
# python train.py --training.lr 0.01

# 3. Context manager (scoped override)
with proto.bind(Training, lr=0.01):
    train()  # Training.lr == 0.01
# Training.lr restored after block

# 4. proto.bind() without context manager
proto.bind(**{"training.lr": 0.01, "model.name": "vit"})
```

### Custom Prefix Name

```python
@proto.prefix("train")  # Custom prefix instead of class name
class TrainingConfig:
    lr: float = 0.001

# CLI: --train.lr 0.01 (not --trainingconfig.lr)
```

### Post-Initialization Hook

```python
@proto.prefix
class Config:
    lr: float = 0.01
    total: int = None

    def __post_init__(self):
        if self.lr > 1:
            raise ValueError("lr too high")
        self.total = int(self.lr * 1000)

c = Config()
print(c.total)  # 10
```

---

## @proto Decorator

Creates multi-instance configuration classes (not singletons).

### Basic Usage

```python
@proto
class OptimizerConfig:
    name: str = "adam"
    lr: float = 0.001

# Create multiple instances
adam = OptimizerConfig(name="adam", lr=0.001)
sgd = OptimizerConfig(name="sgd", lr=0.01)
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

### @proto vs @proto.prefix

| Feature | `@proto.prefix` | `@proto` |
|---------|-----------------|----------|
| Singleton | Yes | No |
| CLI integration | Auto with @proto.cli | Manual |
| Access pattern | `Class.attr` | `instance.attr` |
| Use case | Global config | Reusable templates |

---

## Type Annotations

### Supported Types

| Python Type | CLI Display | Conversion |
|-------------|-------------|------------|
| `int` | `INT` | `"42"` → `42` |
| `float` | `FLOAT` | `"0.01"` → `0.01` |
| `str` | `STR` | Pass through |
| `bool` | `BOOL` | `"true"/"1"/"yes"` → `True` |
| `Enum` | `{MEMBER,...}` | Member name lookup |
| `Literal["a","b"]` | `VALUE` | Validated against choices |
| `List[T]` | `VALUE` | Multiple args collected |
| `Tuple[T, ...]` | `VALUE` | Variable-length tuple |
| `Tuple[int, str]` | `VALUE` | Fixed-length typed tuple |
| `Optional[T]` | `VALUE` | `None` if not provided |
| `Path` | `VALUE` | String → `Path` object |

### Union Types (Subcommand Pattern)

```python
from dataclasses import dataclass

@dataclass
class Train:
    lr: float = 0.001
    epochs: int = 100

@dataclass
class Evaluate:
    checkpoint: str = "model.pt"

@proto.cli
def main(mode: Train | Evaluate):
    if isinstance(mode, Train):
        print(f"Training: lr={mode.lr}")
    else:
        print(f"Evaluating: {mode.checkpoint}")
```

```bash
python main.py train --lr 0.01
python main.py evaluate --checkpoint best.pt
# Also: --mode:train, --mode:Train, --mode:perspective-camera
```

### Enum Types

```python
from enum import Enum

class Optimizer(Enum):
    ADAM = "adam"
    SGD = "sgd"

@proto.cli
def train(optimizer: Optimizer = Optimizer.ADAM): ...
```

```bash
python train.py --optimizer SGD
```

### List and Tuple Types

```python
@proto.cli
def train(
    gpu_ids: List[int] = [0, 1],  # --gpu-ids 0 1 2 3
    dims: Tuple[int, int] = (224, 224),  # --dims 256 256
    scales: Tuple[float, ...] = (0.5, 1.0),  # --scales 0.5 0.75 1.0
): ...
```

---

## Help Text Generation

### From Inline Comments

```python
@proto.cli
def train(
    lr: float = 0.001,  # Learning rate for optimizer
): ...
```

### From Docstrings

```python
@proto.cli
def train(lr: float = 0.001):
    """Train a model.

    Args:
        lr: Learning rate for the optimizer
    """
```

### Combined Output

```
--lr FLOAT    Learning rate for optimizer
              Learning rate for the optimizer (default: 0.001)
```

### Help for @proto.prefix Classes

```
Model options:
  Model configuration.

  --model.name STR     Architecture (default: resnet50)
  --model.dropout FLOAT
                       Dropout rate (default: 0.5)
```
