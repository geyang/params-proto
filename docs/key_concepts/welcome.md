# Welcome to params-proto

**params-proto** is a lightweight Python library that automatically generates command-line interfaces (CLIs) and configuration management from type-annotated functions and classes.

## What Problem Does It Solve?

Instead of manually building argument parsers and config systems, you write simple Python code with type hints—params-proto handles the rest:

```python
from params_proto import proto

@proto.cli
def train(
    learning_rate: float = 0.001,  # Learning rate
    batch_size: int = 32,           # Batch size
    epochs: int = 10,               # Number of epochs
):
    """Train a machine learning model."""
    print(f"Training with lr={learning_rate}, batch={batch_size}")

if __name__ == "__main__":
    train()
```

Run it from the command line:

```bash
$ python train.py --help

usage: train.py [-h] [--learning-rate FLOAT] [--batch-size INT] [--epochs INT]

Train a machine learning model.

options:
  -h, --help             show this help message and exit
  --learning-rate FLOAT  Learning rate (default: 0.001)
  --batch-size INT       Batch size (default: 32)
  --epochs INT           Number of epochs (default: 10)

$ python train.py --learning-rate 0.01 --batch-size 64 --epochs 50
Training with lr=0.01, batch=64
```

## Key Features

✅ **Automatic CLI generation** - No argparse boilerplate needed
✅ **Type-based help text** - Automatic type detection and display
✅ **Configuration as code** - Reusable, composable config objects
✅ **Union types as subcommands** - Choose between multiple configurations
✅ **Environment variables** - Load defaults from env vars
✅ **Hyperparameter sweeps** - Easy parameter exploration
✅ **Singleton prefixes** - Global namespaced configuration groups

## Quick Start Paths

**For scripts and CLI tools:**
Start with [Configuration Patterns](configuration-patterns) to learn function-based configs.

**For libraries and reusable components:**
Start with [Core Concepts](core-concepts) to understand decorators and classes.

**For detailed CLI features:**
Check [CLI Fundamentals](cli-fundamentals) for naming, help generation, and types.

**For advanced patterns:**
See [Advanced Patterns](advanced-patterns) for prefixes, singletons, and composition.

## Next Steps

1. Read [Core Concepts](core-concepts) - Understand the 3 main decorators
2. Pick your pattern:
   - Functions → [Configuration Patterns](configuration-patterns)
   - Classes → [Advanced Patterns](advanced-patterns)
   - Union types → [Union Types](union-types)
3. Explore features:
   - CLI basics → [CLI Fundamentals](cli-fundamentals)
   - CLI patterns → [CLI Patterns](cli-patterns)
   - Environment variables → [Environment Variables](environment-variables)
   - Parameter sweeps → [Hyperparameter Sweeps](hyperparameter-sweeps)

## Documentation

- [Core Concepts](core-concepts) - Decorators and basic usage
- [Configuration Patterns](configuration-patterns) - Functions vs classes
- [CLI Fundamentals](cli-fundamentals) - Building CLIs
- [CLI Patterns](cli-patterns) - Advanced CLI patterns
- [Union Types](union-types) - Subcommands and optional parameters
- [Advanced Patterns](advanced-patterns) - Prefixes and composition
- [Type System](type-system) - Supported types and conversion
- [Environment Variables](environment-variables) - EnvVar integration
- [Parameter Overrides](parameter-overrides) - Multiple override methods
- [Hyperparameter Sweeps](hyperparameter-sweeps) - Systematic exploration
- [Parameter Iteration](parameter-iteration) - Lightweight sweeps with piter

## Features

### Automatic CLI from Functions

```python
@proto.cli
def train(lr: float = 0.001, batch_size: int = 32):
    print(f"Training with lr={lr}, batch_size={batch_size}")
```

→ Automatically generates help, parses args, handles types.

### Configuration Classes

```python
@proto
class Config:
    """Training configuration."""
    lr: float = 0.001
    batch_size: int = 32

@proto.cli
def train(config: Config):
    print(f"Using config: {config}")
```

→ Reusable, composable, type-safe configurations.

### Union Types as Subcommands

```python
@proto.cli
def train(optimizer: Adam | SGD):
    """Choose optimizer and train."""
    pass

# CLI: python train.py --optimizer:Adam --optimizer.lr 0.01
```

### Global Prefixed Configs

```python
@proto.prefix
class Model:
    name: str = "resnet50"

@proto.cli
def main():
    print(Model.name)  # Access globally

# CLI: python main.py --model.name vit
```

## Philosophy

- **Minimal boilerplate** - Focus on logic, not CLI setup
- **Type-driven** - Leverage Python's type system
- **Composable** - Build complex configs from simple pieces
- **CLI-first** - Design for command-line usage
- **Pythonic** - Use familiar Python patterns and idioms
