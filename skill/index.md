---
title: params-proto Skill
description: Claude skill for working with params-proto v3 - declarative hyperparameter management for ML
---

# params-proto Skill

A Claude skill for working with params-proto v3 - a declarative hyperparameter management library for ML.

## When to Use This Skill

Use this skill when helping users:
- Create CLI applications with type-hinted parameters
- Configure ML training scripts with hyperparameters
- Set up multi-namespace configurations
- Work with environment variables in configs
- Create hyperparameter sweeps

## Quick Reference

### Three Decorators

| Decorator | Purpose | Use Case |
|-----------|---------|----------|
| `@proto.cli` | CLI entry point | Script entry points, parses sys.argv |
| `@proto.prefix` | Singleton config | Global namespaced configs (`Model.lr`) |
| `@proto` | Multi-instance | Reusable config classes |

### Basic Pattern

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

### Multi-Namespace Pattern

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
```

CLI: `python train.py --model.name vit --training.lr 0.01`

## Skill Contents

- [quick-reference.md](quick-reference.md) - Cheat sheet for common patterns
- [api/proto-cli.md](api/proto-cli.md) - @proto.cli decorator details
- [api/proto-prefix.md](api/proto-prefix.md) - @proto.prefix for singletons
- [api/types.md](api/types.md) - Supported type annotations
- [features/help-generation.md](features/help-generation.md) - Auto help text
- [features/environment-vars.md](features/environment-vars.md) - EnvVar support
- [features/sweeps.md](features/sweeps.md) - Hyperparameter sweeps
- [examples/patterns.md](examples/patterns.md) - Common patterns
