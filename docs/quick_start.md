# Quick Start Guide

Get started with params-proto v3 in 10 minutes!

```bash
pip install params-proto=={VERSION}
```

Or with uv:

```bash
uv add params-proto=={VERSION}
```

## For AI Assistants

We provide a Claude skill for params-proto. To use it, either type in Claude Code:

```
# add https://raw.githubusercontent.com/geyang/params-proto/main/skill/index.md as a skill
```

Or add this import to your project's `CLAUDE.md` file:

```markdown
@import https://raw.githubusercontent.com/geyang/params-proto/main/skill/index.md
```

## Your First Proto Function CLI

Let's start with a simple train function---Just add `@proto.cli` and params-proto handles the rest.

Create `train.py`:

```python
from params_proto import proto


@proto.cli
def train(
  lr: float = 0.001,  # Learning rate
  batch_size: int = 32,  # Training batch size
  epochs: int = 100,  # Number of training epochs
):
  """Train a neural network on CIFAR-10."""
  print(f"Training: {epochs} epochs, lr={lr}, batch_size={batch_size}")


if __name__ == "__main__":
  train()
```

**That's it!** No argparse boilerplate. Just type hints and inline comments.

```{ansi-block}
:string_escape:

\x1b[1mTraining: 100 epochs, lr=0.001, batch_size=32\x1b[0m
```

## Overview: Three Decorators

params-proto v3 provides three decorators in two categories:

### Config Decorators (define parameter schemas)

| Decorator       | Scope              | Use Case                                                 |
|-----------------|--------------------|----------------------------------------------------------|
| `@proto`        | Multiple instances | Library code, reusable components                        |
| `@proto.prefix` | Singleton (global) | Namespaced config groups (`Model.lr`, `Training.epochs`) |

### App Decorator (creates CLI entry point)

| Decorator    | Use Case                                                        |
|--------------|-----------------------------------------------------------------|
| `@proto.cli` | Script entry points—wraps a function or class to parse CLI args |

**Typical pattern:** Define configs with `@proto`/`@proto.prefix`, then create an entry point with `@proto.cli`.

Run it:

```bash
$ python train.py --help
```

```{ansi-block}
:string_escape:

usage: train.py [-h] [--lr \x1b[1m\x1b[94mFLOAT\x1b[0m] [--batch-size \x1b[1m\x1b[94mINT\x1b[0m] [--epochs \x1b[1m\x1b[94mINT\x1b[0m]

Train a neural network on CIFAR-10.

options:
  -h, --help           show this help message and exit
  --lr \x1b[1m\x1b[94mFLOAT\x1b[0m           Learning rate \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m0.001\x1b[0m\x1b[36m)\x1b[0m
  --batch-size \x1b[1m\x1b[94mINT\x1b[0m     Training batch size \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m32\x1b[0m\x1b[36m)\x1b[0m
  --epochs \x1b[1m\x1b[94mINT\x1b[0m         Number of training epochs \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m100\x1b[0m\x1b[36m)\x1b[0m
```

```bash
$ python train.py
```

```{ansi-block}
:string_escape:

\x1b[1mTraining for 100 epochs\x1b[0m
  Learning rate: 0.001
  Batch size: 32
```

```bash
$ python train.py --lr 0.01 --batch-size 64
```

```{ansi-block}
:string_escape:

\x1b[1mTraining for 100 epochs\x1b[0m
  Learning rate: 0.01
  Batch size: 64
```

## Python Namespaces as Config Singletons

As your config grows, classes provide better organization. Use `@proto` to turn a class into a configuration.

Create `config.py`:

```python
from params_proto import proto


@proto
class Params:
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


# Access as class attributes
print(f"Training {Params.model} with lr={Params.lr}")

# Or create an instance
config = Params()
print(f"Batch size: {config.batch_size}")
```

**Key difference**: Classes use attributes (`Params.lr`), functions use parameters (`train(lr=0.001)`).

## Composing Configurations from Multiple Locations

For larger projects, split configuration into logical groups using `@proto.prefix`. This creates namespaced parameters
like `--model.name` and `--training.lr`.

Create `train_rl.py`:

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


if __name__ == "__main__":
  main()
```

Notice how parameters are organized into groups! Run it:

```{ansi-block}
:string_escape:

usage: train_rl.py [-h] [--seed \x1b[1m\x1b[94mINT\x1b[0m] [--device \x1b[1m\x1b[94mSTR\x1b[0m] [OPTIONS]

Train a model on a dataset.

options:
  -h, --help                   show this help message and exit
  --seed \x1b[1m\x1b[94mINT\x1b[0m                   Random seed \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m42\x1b[0m\x1b[36m)\x1b[0m
  --device \x1b[1m\x1b[94mSTR\x1b[0m                 Device to use (cuda/cpu) \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36mcuda\x1b[0m\x1b[36m)\x1b[0m

Model options:
  Model configuration.

  --model.name \x1b[1m\x1b[94mSTR\x1b[0m             Architecture name \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36mresnet50\x1b[0m\x1b[36m)\x1b[0m
  --model.pretrained           Use pretrained weights \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36mTrue\x1b[0m\x1b[36m)\x1b[0m
  --model.dropout \x1b[1m\x1b[94mFLOAT\x1b[0m        Dropout rate \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m0.5\x1b[0m\x1b[36m)\x1b[0m

Data options:
  Data configuration.

  --data.dataset \x1b[1m\x1b[94mSTR\x1b[0m           Dataset name \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36mcifar10\x1b[0m\x1b[36m)\x1b[0m
  --data.data-dir \x1b[1m\x1b[94mSTR\x1b[0m          Data directory \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m./data\x1b[0m\x1b[36m)\x1b[0m
  --data.num-workers \x1b[1m\x1b[94mINT\x1b[0m       Data loading workers \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m4\x1b[0m\x1b[36m)\x1b[0m

Training options:
  Training hyperparameters.

  --training.lr \x1b[1m\x1b[94mFLOAT\x1b[0m          Learning rate \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m0.001\x1b[0m\x1b[36m)\x1b[0m
  --training.batch-size \x1b[1m\x1b[94mINT\x1b[0m    Batch size \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m32\x1b[0m\x1b[36m)\x1b[0m
  --training.epochs \x1b[1m\x1b[94mINT\x1b[0m        Number of epochs \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m100\x1b[0m\x1b[36m)\x1b[0m
```

Run it:

```{ansi-block}
:string_escape:

\x1b[1mTraining resnet50 on cifar10\x1b[0m
  LR: 0.001, Batch size: 32
  Device: cuda, Seed: 42
```

With custom arguments:

```{ansi-block}
:string_escape:

\x1b[1mTraining vit on cifar10\x1b[0m
  LR: 0.0001, Batch size: 32
  Device: cuda, Seed: 123
```

**Prefixes create organization**: Each `@proto.prefix` class becomes a group in the help text and uses dotted notation
on the CLI.

## Summary: Three Patterns

You've learned three core patterns in 10 minutes:

| Pattern       | Decorator       | Use Case                | Access Pattern                  |
|---------------|-----------------|-------------------------|---------------------------------|
| **Functions** | `@proto.cli`    | Simple CLIs, scripts    | Function parameters             |
| **Classes**   | `@proto`        | Organized configs       | Class attributes                |
| **Prefixes**  | `@proto.prefix` | Multi-namespace configs | Namespaced (e.g., `Model.name`) |

## Quick Tips

**Type hints are required**: params-proto uses type hints for parsing and validation.

```python
lr: float = 0.001  # ✅ Required
lr = 0.001  # ❌ Won't work
```

**Inline comments become help text**: Write comments after parameters.

```python
lr: float = 0.001  # Learning rate  ← This appears in --help
```

**Override anywhere**: CLI, direct assignment, context managers, config files.

```python
# CLI
$ python
train.py - -lr
0.01

# Direct assignment
Training.lr = 0.01

# Context manager
with proto.bind(lr=0.01):
  train()
```

## What's Next?

Now that you've mastered the basics, dive deeper:

- **[Decorators](key_concepts/decorators.md)** - `@proto.cli`, `@proto.prefix`, `@proto` in detail
- **[Functions](key_concepts/functions.md)** - Function-based CLIs and advanced patterns
- **[Classes](key_concepts/classes.md)** - Class-based configuration strategies
- **[Types](key_concepts/types.md)** - Union types, Literals, Enums, and validation
- **[Prefixes](key_concepts/prefixes.md)** - Composing configs from multiple namespaces
- **[Overrides](key_concepts/overrides.md)** - CLI, programmatic, YAML, and more
- **[Subcommands](key_concepts/subcommands.md)** - Multi-command CLIs (design doc)

Or jump straight to **[Examples](examples/)** for real-world patterns!
