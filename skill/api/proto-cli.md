---
title: "@proto.cli Decorator"
description: Create CLI entry points from functions or classes
---

# @proto.cli Decorator

The `@proto.cli` decorator creates CLI entry points from functions or classes.

## Basic Usage

```python
from params_proto import proto

@proto.cli
def train(
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Batch size
    epochs: int = 100,  # Training epochs
):
    """Train a neural network."""
    print(f"Training with lr={lr}, batch={batch_size}, epochs={epochs}")

if __name__ == "__main__":
    train()
```

## Parameters

### `prog` - Override Program Name

```python
@proto.cli(prog="my-trainer")
def train(lr: float = 0.001): ...
```

Help shows `usage: my-trainer` instead of filename.

## CLI Argument Parsing

### Named Arguments

```bash
python train.py --lr 0.01 --batch-size 64
```

- Underscores in Python → hyphens in CLI
- `learning_rate` → `--learning-rate`

### Positional Arguments (Required Params)

```python
@proto.cli
def train(
    seed: int,  # Required - no default
    lr: float = 0.001,
): ...
```

```bash
python train.py 42  # seed as positional
python train.py --seed 42  # or named
```

### Boolean Flags

```python
@proto.cli
def train(
    verbose: bool = False,  # Use --verbose to enable
    cuda: bool = True,  # Use --no-cuda to disable
): ...
```

- `default=False` → `--flag` sets True
- `default=True` → `--no-flag` sets False

Help text shows appropriate form:
```
--verbose BOOL       Enable verbose (default: False)
--no-cuda BOOL       Use CUDA (default: True)
```

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

### Combined (Multi-line)

```python
@proto.cli
def train(
    lr: float = 0.001,  # Learning rate
):
    """Train a model.

    Args:
        lr: Controls gradient step size
    """
```

Output:
```
--lr FLOAT    Learning rate
              Controls gradient step size (default: 0.001)
```

## Programmatic Calls

When called with arguments, CLI parsing is bypassed:

```python
@proto.cli
def train(lr: float = 0.001): ...

# These bypass sys.argv parsing:
train(lr=0.01)
train(0.01)
```

## With @proto.prefix Classes

`@proto.cli` automatically picks up `@proto.prefix` singletons:

```python
@proto.prefix
class Model:
    name: str = "resnet"

@proto.cli
def train(seed: int = 42):
    print(f"Training {Model.name}")

if __name__ == "__main__":
    train()
```

```bash
python train.py --model.name vit --seed 123
```

## Error Handling

### Missing Required Arguments

```bash
$ python train.py
error: the following argument is required: seed
```

### Invalid Values

```bash
$ python train.py --lr not-a-number
error: invalid value for --lr: not-a-number
```

### Unknown Arguments

```bash
$ python train.py --unknown-arg 123
error: unrecognized argument: --unknown-arg
```

## Exit Codes

- `0` - Success or `--help`
- `1` - Argument parsing error
- Other - From your function

## With @classmethod and @staticmethod

Place `@proto.cli` on the OUTSIDE (applied last):

```python
class Trainer:
    @proto.cli          # OUTSIDE - receives the descriptor
    @staticmethod
    def evaluate(model_path: str, threshold: float = 0.5):
        """Evaluate a model."""
        pass

    @proto.cli          # OUTSIDE - receives the descriptor
    @classmethod
    def train(cls, lr: float = 0.01):
        """Train using class configuration."""
        return cls.run(lr)
```

**Why this order?** Python applies decorators bottom-up. `@proto.cli` on the outside
receives the `classmethod`/`staticmethod` descriptor, allowing it to:
- Detect the method type
- Exclude `cls`/`self` from CLI parameters automatically
- Handle method binding correctly when called

**Wrong order** (proto inside) would cause `cls` to appear as a CLI argument.

## Best Practices

1. **Always use type hints** - Required for CLI parsing
2. **Add inline comments** - Become help text
3. **Use descriptive docstrings** - Become CLI description
4. **Group related params** - Use `@proto.prefix` for organization
5. **Test with `--help`** - Verify help text is clear
6. **Decorator order for methods** - Put `@proto.cli` on the OUTSIDE of `@classmethod`/`@staticmethod`
