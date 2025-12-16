# params-proto: Modern Declarative Parameters for Machine Learning

[![Documentation Status](https://readthedocs.org/projects/params-proto/badge/?version=latest)](https://params-proto.readthedocs.io/en/latest/?badge=latest)
[![GitHub Release](https://img.shields.io/github/release/geyang/params-proto.svg)](https://github.com/geyang/params-proto/releases)
[![PyPI version](https://badge.fury.io/py/params-proto.svg)](https://badge.fury.io/py/params-proto)

**params-proto** is a lightweight, decorator-based library for defining configurations in Python. Write your parameters once with type hints and inline documentation, and get automatic CLI parsing, validation, and help generation.

> **Note**: This is v3 with a completely redesigned API. For the v2 API, see [params-proto-v2](https://github.com/geyang/params-proto-v2).

## Why params-proto?

**Stop fighting with argparse and click.** With params-proto, your configuration **is** your documentation:

```python
from params_proto import proto

@proto.cli
def train_mnist(
    batch_size: int = 128,  # Training batch size
    lr: float = 0.001,  # Learning rate
    epochs: int = 10,  # Number of training epochs
):
    """Train an MLP on MNIST dataset."""
    print(f"Training with lr={lr}, batch_size={batch_size}, epochs={epochs}")
    # Your training code here...

if __name__ == "__main__":
    train_mnist()
```

**That's it!** No argparse boilerplate, no manual help strings, no type conversion logic. Just pure Python functions with type hints and inline comments.

Run it:
```bash
$ python train.py --help
usage: train.py [-h] [--batch-size INT] [--lr FLOAT] [--epochs INT]

Train an MLP on MNIST dataset.

options:
  -h, --help           show this help message and exit
  --batch-size INT     Training batch size (default: 128)
  --lr FLOAT           Learning rate (default: 0.001)
  --epochs INT         Number of training epochs (default: 10)

$ python train.py --lr 0.01 --batch-size 256
Training with lr=0.01, batch_size=256, epochs=10
```

> **Note**: The actual terminal output includes beautiful ANSI colors! See the demo below or check the [documentation](https://params-proto.readthedocs.io/) for colorized examples.

## Try It Now

Want to see the colorized help in action? Clone the repo and run the demo:

```bash
# Clone and setup
git clone https://github.com/geyang/params-proto.git
cd params-proto
uv sync

# See the colorized help (with bright blue types, bold cyan defaults, bold red required)
uv run python scratch/demo_v3.py --help

# Try running without required --seed (shows error)
uv run python scratch/demo_v3.py
# Error: the following arguments are required: --seed

# Run with required parameter (keyword syntax)
uv run python scratch/demo_v3.py --seed 42

# Or use positional syntax
uv run python scratch/demo_v3.py 42
```

## Installation

```bash
pip install params-proto==3.0.0-rc7
```

## Key Features

### 1. Function-based Configs
Define parameters using type-annotated functions:

```python
@proto.cli
def train(
    model: str = "resnet50",  # Model architecture
    dataset: str = "imagenet",  # Dataset to use
    gpu: bool = True,  # Enable GPU acceleration
):
    """Train a model on a dataset."""
    print(f"Training {model} on {dataset}")
```

### 2. Class-based Configs
Or use classes for more structure:

```python
@proto
class Params:    """Training configuration."""

    # Model settings
    model: str = "resnet50"
    pretrained: bool = True  # Use pretrained weights

    # Training settings
    lr: float = 0.001  # Learning rate
    batch_size: int = 32  # Batch size
    epochs: int = 100  # Number of epochs
```

### 3. Singleton Prefixed Configs
Create modular, reusable configuration groups:

```python
from params_proto import proto

@proto.prefix
class Environment:
    """Environment configuration."""
    domain: str = "cartpole"  # Domain name
    task: str = "swingup"  # Task name
    time_limit: float = 10.0  # Episode time limit

@proto.prefix
class Agent:
    """Agent hyperparameters."""
    algorithm: str = "SAC"  # RL algorithm
    lr: float = 3e-4  # Learning rate
    gamma: float = 0.99  # Discount factor

@proto.cli
def train_rl(
    seed: int = 0,  # Random seed
    total_steps: int = 1000000,  # Total training steps
):
    """Train RL agent on dm_control."""
    print(f"Training {Agent.algorithm} on {Environment.domain}-{Environment.task}")
    print(f"Agent LR: {Agent.lr}, Gamma: {Agent.gamma}")
```

Command line:
```bash
$ python train_rl.py --Agent.lr 0.001 --Environment.domain walker --seed 42
Training SAC on walker-swingup
Agent LR: 0.001, Gamma: 0.99
```

### 4. Multiple Override Patterns

Override parameters in multiple ways:

```python
# 1. Command line
$ python train.py --lr 0.01

# 2. Direct attribute assignment
Params.lr = 0.01

# 3. Function call with kwargs
train(lr=0.01, batch_size=256)

# 4. Using proto.bind() context manager
with proto.bind(lr=0.01, **{"train.epochs": 50}):
    train()
```

### 5. Rich Type System

Support for complex types:

```python
from typing import Literal, Union
from enum import Enum, auto

class Optimizer(Enum):
    ADAM = auto()
    SGD = auto()
    RMSPROP = auto()

@proto
class Params:    # Union types
    precision: Literal["fp16", "fp32", "fp64"] = "fp32"

    # Enums
    optimizer: Optimizer = Optimizer.ADAM

    # Tuples
    image_size: tuple[int, int] = (224, 224)

    # Optional types
    checkpoint: str | None = None
```

## Quick Start

1. **Define your configuration** with a decorated function or class
2. **Add type hints** for automatic validation
3. **Add inline comments** for automatic documentation
4. **Call your function** - params-proto handles the rest!

See our [Quick Start Guide](https://params-proto.readthedocs.io/en/latest/quick_start.html) for more.

## Documentation

- **[Quick Start](https://params-proto.readthedocs.io/en/latest/quick_start.html)** - Get started in 5 minutes
- **[User Guide](https://params-proto.readthedocs.io/en/latest/guide/)** - Detailed documentation
- **[Examples](https://params-proto.readthedocs.io/en/latest/examples/)** - Real-world usage patterns
- **[API Reference](https://params-proto.readthedocs.io/en/latest/api/)** - Complete API documentation
- **[Migration from v2](https://params-proto.readthedocs.io/en/latest/migration.html)** - Upgrade guide

## What Changed in v3?

**v2 (old)**: Class-based with inheritance
```python
from params_proto import ParamsProto

class Args(ParamsProto):
    lr = 0.001
    batch_size = 32
```

**v3 (new)**: Decorator-based with type hints
```python
from params_proto import proto

@proto
class Args:
    lr: float = 0.001  # Learning rate
    batch_size: int = 32  # Batch size
```

Key improvements:
- ✅ Cleaner decorator syntax (no inheritance needed)
- ✅ Full IDE support with type hints
- ✅ Inline documentation becomes automatic help text
- ✅ Support for functions, not just classes
- ✅ Better Union types and Enum support
- ✅ Simplified singleton pattern with `@proto.prefix`

## Contributing

```bash
git clone https://github.com/episodeyang/params_proto.git
cd params_proto
make dev test
```

To publish:
```bash
make publish
```

## License

MIT License - see [LICENSE](LICENSE) for details.
