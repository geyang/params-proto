# params-proto: Composite Singleton Hyper-Parameter Management

**params-proto v3**
is a declarative hyperparameter management library for machine learning. Write your parameters once with type hints and
inline comments to get automatic CLI parsing, help generation, and declarative parameter sweeps with explicit error
messages.

- Automatically parse type hints and inline comments into an CLI program
- Your IDE will provide autocompletion and type checking for your parameters
- As simple as a class namespace or a function, progressively build up to more complex programs.
- Multiple override patterns: CLI, direct assignment, context managers, and yaml config files

## Quick Start

Install params-proto using uv or pip:

```shell
uv add params-proto
```

or

```shell
pip install params-proto
```

You can convert this python funciton into a cli program with a single decorator:

```python
from params_proto import proto


@proto.cli
def train(
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Training batch size
    n_epochs: int = 100,  # Number of training epochs
):
    """Train a neural network on CIFAR-10 dataset.

    This function trains a ResNet model using the specified hyperparameters.
    Training progress and metrics are logged to stdout.
    """
    print(f"Training with lr={lr}, batch_size={batch_size}, n_epochs={n_epochs}")
    # Your training code here...


if __name__ == "__main__":
    train()
```

Now running it from the command line gives you automatic help:

```shell
$ python train.py --help
usage: train.py [-h] [--lr FLOAT] [--batch-size INT] [--n-epochs INT]

Train a neural network on CIFAR-10 dataset.

This function trains a ResNet model using the specified hyperparameters.
Training progress and metrics are logged to stdout.

options:
  -h, --help           show this help message and exit
  --lr FLOAT           Learning rate (default: 0.001)
  --batch-size INT     Training batch size (default: 32)
  --n-epochs INT       Number of training epochs (default: 100)
```

And you can override parameters:

```shell
$ python train.py --lr 0.01 --batch-size 64
Training with lr=0.01, batch_size=64, n_epochs=100
```

## Composing Multiple Singleton Schemas 

```python
from params_proto import proto


@proto.prefix
class Model:
    """Model configuration."""
    name: str = "resnet50"  # Model architecture
    pretrained: bool = True  # Use pretrained weights


@proto.prefix
class Train:
    """Training hyperparameters."""
    lr: float = 0.001  # Learning rate
    batch_size: int = 32  # Batch size


@proto.cli
def main(seed: int = 42):  # Random seed
    """Train a model."""
    print(f"Training {Model.name} with lr={Train.lr}")


if __name__ == "__main__":
    # this launches the cli program.
    main()
```

Command line usage:

```bash
$ python train.py --model.name vit --train.lr 0.0001
```

## Documentation Contents

```{toctree}
:maxdepth: 1
:caption: Getting Started

quick_start
migration
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guide/decorators
guide/functions
guide/classes
guide/types
guide/overrides
guide/prefixes
```

```{toctree}
:maxdepth: 1
:caption: Examples

examples/basic_usage
examples/ml_training
examples/rl_agent
examples/cli_applications
```

```{toctree}
:maxdepth: 1
:caption: API Reference

api/proto
api/utils
```

```{toctree}
:maxdepth: 1
:caption: Additional Resources

release_notes
```

## What Problem Does This Solve?

"Experiment Parameter Hell" occurs when you have numerous parameters scattered across your ML codebase, making it hard
to:

- Track what parameters exist
- Get IDE autocompletion and type checking
- Change parameters from the command line
- Maintain parameter documentation

params-proto solves this by providing a declarative way to define parameters that integrates seamlessly with Python IDEs
and command-line interfaces.

## Why v3?

params-proto v3 is a complete redesign focused on simplicity and modern Python:

| Feature     | v2                | v3           |
|-------------|-------------------|--------------|
| API Style   | Class inheritance | Decorators   |
| Type Hints  | Optional          | Required     |
| Inline Docs | Manual            | Automatic    |
| Functions   | Not supported     | Full support |
| Union Types | Limited           | Full support |
| IDE Support | Basic             | Excellent    |

See the [Migration Guide](migration.md) for upgrading from v2.

## GitHub Repository

The source code is available on [GitHub](https://github.com/geyang/params-proto).

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
