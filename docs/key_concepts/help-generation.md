# Help Text Generation

params-proto automatically extracts and generates help text from your code documentation.

## Overview

Help text is generated from multiple sources in a specific order:

1. **Inline comments** - Quick parameter description
2. **Docstring Args section** - Detailed parameter documentation
3. **Function docstring** - Overall CLI description
4. **Auto-generated descriptions** - If no documentation provided

## Inline Comments

The simplest form of documentation:

```python
@proto.cli
def train(
  lr: float = 0.001,  # This becomes the CLI help text
  batch_size: int = 32,  # Keep it short and descriptive
):
  pass
```

**Generated help:**

```
--lr FLOAT           This becomes the CLI help text (default: 0.001)
--batch-size INT     Keep it short and descriptive (default: 32)
```

## Docstring Args Section

For more detailed documentation, use the Args section in your docstring:

```python
@proto.cli
def train(
  lr: float = 0.001,  # Learning rate
):
  """Train a model.

  Args:
      lr: Learning rate for the optimizer. Typical values are 0.001 for
          Adam and 0.01-0.1 for SGD. Reduce if training is unstable.
  """
  pass
```

**Generated help combines both:**

```
--lr FLOAT           Learning rate
                     Learning rate for the optimizer. Typical values are 0.001
                     for Adam and 0.01-0.1 for SGD. Reduce if training is
                     unstable. (default: 0.001)
```

The inline comment appears first, then details from the docstring Args section.

## Function Docstring

The function's main docstring becomes the CLI description:

```python
@proto.cli
def train(lr: float = 0.001):
  """Train a neural network on CIFAR-10.

  This function implements the full training loop including
  data loading, forward/backward passes, and checkpointing.
  """
  pass
```

**Generated help:**

```
usage: train.py [-h] [--lr FLOAT]

Train a neural network on CIFAR-10.

options:
  ...
```

All text **before** the first section header (`Args:`, `Returns:`, `Raises:`, etc.) appears in the help text. This allows multi-paragraph descriptions as long as they come before any structured documentation sections.

## Auto-Generated Descriptions

If no documentation is provided, params-proto generates basic descriptions from parameter names:

```python
@proto.cli
def train(
  learning_rate: float = 0.001,  # No comment
  batch_size: int = 32,  # No comment
):
  """Train a model."""
  pass
```

**Generated help:**

```
--learning-rate FLOAT  Learning rate (default: 0.001)
--batch-size INT       Batch size (default: 32)
```

The parameter name is converted from `snake_case` to a readable description.

## Best Practices

### 1. Use Inline Comments for Quick Reference

```python
# ✓ Good: inline comments for quick reference
@proto.cli
def train(
  lr: float = 0.001,  # Learning rate
  batch_size: int = 32,  # Training batch size
  epochs: int = 100,  # Number of epochs
):
  pass
```

### 2. Add Docstring for Details

```python
# ✓ Good: docstring for comprehensive documentation
@proto.cli
def train(
  lr: float = 0.001,  # Learning rate
):
  """Train a neural network.

  Args:
      lr: Learning rate for the optimizer. Typical values are 0.001 for
          Adam and 0.01-0.1 for SGD. Reduce if training is unstable.
  """
  pass
```

### 3. Use Type Hints for Constraints

```python
# ✓ Good: Literal types document valid values
from typing import Literal

@proto.cli
def train(
  optimizer: Literal["adam", "sgd", "rmsprop"] = "adam",  # Optimizer type
):
  """Train with specific optimizers only."""
  pass
```

### 4. Document Class Attributes

```python
# ✓ Good: Document configuration classes
@proto
class TrainConfig:
  """Training configuration.

  Attributes:
      lr: Learning rate for the optimizer
      batch_size: Number of samples per batch
      epochs: Number of training epochs
  """
  lr: float = 0.001
  batch_size: int = 32
  epochs: int = 100
```

### 5. Multi-Paragraph Descriptions

```python
# ✓ Good: Multi-paragraph docstring
@proto.cli
def train(lr: float = 0.001, batch_size: int = 32):
  """Train a neural network.

  This is the main training loop that handles:
  - Data loading and preprocessing
  - Model forward and backward passes
  - Gradient updates and optimization
  - Checkpoint saving and resuming

  Typical usage:

      python train.py --lr 0.01 --batch-size 64

  Args:
      lr: Learning rate for the optimizer
      batch_size: Number of samples per batch
  """
  pass
```

Everything before `Args:` appears in the help text.

## Accessing Help Text

You can access the generated help text programmatically:

```python
@proto.cli
def train(lr: float = 0.001):
  """Train a model."""
  pass

# Plain text (for testing, logs, pipes)
print(train.__help_str__)

# Colorized (for terminal display)
print(train.__ansi_str__)
```

This is useful for:
- Testing help generation
- Documentation generation
- Logging and debugging
- Building documentation pages

## Type Hints for Documentation

Use type hints to document valid values:

```python
from typing import Literal
from enum import Enum

# Literal types document allowed values
@proto.cli
def process(
  mode: Literal["train", "eval", "test"] = "train",
):
  """Process data."""
  pass

# Enum types show available options
class Optimizer(Enum):
  ADAM = "adam"
  SGD = "sgd"
  RMSPROP = "rmsprop"

@proto.cli
def optimize(optimizer: Optimizer = Optimizer.ADAM):
  """Optimize model."""
  pass
```

## Related

- [CLI Fundamentals](cli-fundamentals) - Core CLI features
- [CLI Patterns](cli-patterns) - Advanced CLI patterns
- [Naming Conventions](naming-conventions) - Name conversion rules
- [ANSI Formatting](ansi-formatting) - Terminal colors and formatting
- [Configuration Patterns](configuration-patterns) - Function and class documentation
