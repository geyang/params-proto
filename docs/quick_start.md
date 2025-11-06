# Quick Start Guide

Get started with params-proto v3 in 5 minutes!

## Installation

```bash
pip install params-proto
```

Or with uv:
```bash
uv add params-proto
```

## Your First Configuration

Let's create a simple training script with params-proto.

### Step 1: Define Your Parameters

Create a file called `train.py`:

```python
from params_proto import proto

@proto.cli
def train_model(
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Training batch size
    epochs: int = 100,  # Number of training epochs
    model: str = "resnet50",  # Model architecture
):
    """Train a neural network on CIFAR-10."""
    print(f"Training {model} for {epochs} epochs")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")

    # Your training code goes here
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # ... training logic ...

if __name__ == "__main__":
    train_model()
```

That's it! No argparse boilerplate, no manual help strings. Just a normal Python function with type hints and inline comments.

### Step 2: Run from Command Line

Get automatic help:
```bash
$ python train.py --help
usage: train.py [-h] [--lr FLOAT] [--batch-size INT] [--epochs INT] [--model STR]

Train a neural network on CIFAR-10.

options:
  -h, --help           show this help message and exit
  --lr FLOAT           Learning rate (default: 0.001)
  --batch-size INT     Training batch size (default: 32)
  --epochs INT         Number of training epochs (default: 100)
  --model STR          Model architecture (default: resnet50)
```

Run with default values:
```bash
$ python train.py
Training resnet50 for 100 epochs
  Learning rate: 0.001
  Batch size: 32
```

Override parameters:
```bash
$ python train.py --lr 0.01 --batch-size 64 --epochs 50
Training resnet50 for 50 epochs
  Learning rate: 0.01
  Batch size: 64
```

## Using Classes

Prefer classes? No problem:

```python
from params_proto import proto

@proto
class Config:
    """Training configuration."""

    # Model settings
    model: str = "resnet50"  # Model architecture
    pretrained: bool = True  # Use pretrained weights

    # Training settings
    lr: float = 0.001  # Learning rate
    batch_size: int = 32  # Batch size
    epochs: int = 100  # Number of epochs

    # Data settings
    data_dir: str = "./data"  # Data directory
    num_workers: int = 4  # Number of data loading workers

# Create instance
config = Config()
print(f"Training {config.model} with lr={config.lr}")
```

## Modular Configurations

For larger projects, split your configuration into logical groups:

```python
from params_proto import proto

@proto.prefix
class Model:
    """Model configuration."""
    name: str = "resnet50"  # Architecture name
    pretrained: bool = True  # Use pretrained weights
    dropout: float = 0.5  # Dropout rate

@proto.prefix
class Data:
    """Data configuration."""
    dataset: str = "cifar10"  # Dataset name
    data_dir: str = "./data"  # Data directory
    num_workers: int = 4  # Data loading workers

@proto.prefix
class Training:
    """Training hyperparameters."""
    lr: float = 0.001  # Learning rate
    batch_size: int = 32  # Batch size
    epochs: int = 100  # Number of epochs

@proto.cli
def main(
    seed: int = 42,  # Random seed
    device: str = "cuda",  # Device to use (cuda/cpu)
):
    """Train a model on a dataset."""
    print(f"Training {Model.name} on {Data.dataset}")
    print(f"  LR: {Training.lr}, Batch size: {Training.batch_size}")
    print(f"  Device: {device}, Seed: {seed}")

    # Your training code here

if __name__ == "__main__":
    main()
```

Run with prefixed arguments:
```bash
$ python main.py --help
usage: main.py [-h] [--seed INT] [--device STR] [OPTIONS]

Train a model on a dataset.

options:
  -h, --help                   show this help message and exit
  --seed INT                   Random seed (default: 42)
  --device STR                 Device to use (cuda/cpu) (default: cuda)

Model options:
  Model configuration.

  --Model.name STR             Architecture name (default: resnet50)
  --Model.pretrained           Use pretrained weights (default: True)
  --Model.dropout FLOAT        Dropout rate (default: 0.5)

Data options:
  Data configuration.

  --Data.dataset STR           Dataset name (default: cifar10)
  --Data.data-dir STR          Data directory (default: ./data)
  --Data.num-workers INT       Data loading workers (default: 4)

Training options:
  Training hyperparameters.

  --Training.lr FLOAT          Learning rate (default: 0.001)
  --Training.batch-size INT    Batch size (default: 32)
  --Training.epochs INT        Number of epochs (default: 100)

$ python main.py --Model.name vit --Training.lr 0.0001 --seed 123
Training vit on cifar10
  LR: 0.0001, Batch size: 32
  Device: cuda, Seed: 123
```

## Override Patterns

There are multiple ways to override parameters:

### 1. Command Line (shown above)
```bash
python train.py --lr 0.01 --batch-size 64
```

### 2. Direct Attribute Assignment
```python
Config.lr = 0.01
Config.batch_size = 64
```

### 3. Function Call with kwargs
```python
train_model(lr=0.01, batch_size=64)
```

### 4. Using `proto.bind()`
```python
# Context manager (scoped overrides)
with proto.bind(lr=0.01, batch_size=64):
    train_model()

# Direct call (global overrides)
proto.bind(lr=0.01, batch_size=64)
train_model()
```

## Type Annotations

params-proto supports rich type annotations:

```python
from typing import Literal
from enum import Enum, auto

class Optimizer(Enum):
    ADAM = auto()
    SGD = auto()
    RMSPROP = auto()

@proto
class Config:
    # Union types (Python 3.10+)
    precision: Literal["fp16", "fp32", "fp64"] = "fp32"

    # Enums
    optimizer: Optimizer = Optimizer.ADAM

    # Tuples
    image_size: tuple[int, int] = (224, 224)

    # Optional types
    checkpoint: str | None = None
```

## What's Next?

Now that you've got the basics, explore:

- **[User Guide](guide/decorators.md)** - Deep dive into decorators, types, and patterns
- **[Examples](examples/)** - Real-world usage examples
- **[API Reference](api/proto.md)** - Complete API documentation

## Common Patterns

### ML Training Script
```python
@proto.prefix
class Model:
    name: str = "resnet50"
    pretrained: bool = True

@proto.prefix
class Optimizer:
    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0001

@proto.cli
def train(epochs: int = 100, seed: int = 42):
    """Train a model."""
    # Your code here
```

### Experiment Sweeps
```python
# Override in Python code
for lr in [0.001, 0.01, 0.1]:
    Config.lr = lr
    with proto.bind(**{"Model.name": "vit"}):
        train()
```

### Config Files
```python
import json

# Load from JSON
with open("config.json") as f:
    config = json.load(f)

proto.bind(**config)
main()
```

Ready to learn more? Continue to the [User Guide](guide/decorators.md)!
