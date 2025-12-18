# proto Module API Reference

The `proto` module provides the core decorators and functions for params-proto v3.

## Decorators

### `@proto`

Base decorator for classes and functions.

```python
from params_proto import proto

@proto
class Params:
    lr: float = 0.001
    batch_size: int = 32
```

**Usage:**
- Works with both classes and functions
- Requires type annotations
- Enables attribute-based overrides

### `@proto.cli`

Decorator for CLI entry points with automatic help generation.

```python
@proto.cli
def main(
    lr: float = 0.001,  # Learning rate
    epochs: int = 100,  # Number of epochs
):
    """Train a model."""
    pass
```

**Features:**
- Automatic CLI argument parsing
- Help text generation from inline comments
- Support for all standard types

**CLI Arguments:**
- `--param-name VALUE` - Set parameter value
- `--flag` / `--no-flag` - Boolean flags
- `-h` / `--help` - Show help message

### `@proto.prefix`

Decorator for singleton configuration groups with CLI prefixes.

```python
@proto.prefix
class Model:
    """Model configuration."""
    name: str = "resnet50"
```

**Features:**
- Creates singleton instances
- Automatic CLI prefix (`--Model.name value`)
- Global access via class attributes

## Using with Class Methods

The `@proto` decorators work with `@classmethod` and `@staticmethod`. **Important:** Place the proto decorator on the OUTSIDE (applied last).

### Staticmethod

```python
class Trainer:
    @proto.cli          # proto.cli on OUTSIDE
    @staticmethod
    def evaluate(model_path: str, threshold: float = 0.5):
        """Evaluate a saved model."""
        return evaluate_model(model_path, threshold)

# Usage:
Trainer.evaluate("model.pt")
# Or from CLI: python script.py --model-path model.pt --threshold 0.7
```

### Classmethod

```python
class Trainer:
    default_lr = 0.001

    @proto.cli          # proto.cli on OUTSIDE
    @classmethod
    def train(cls, lr: float = 0.01, epochs: int = 100):
        """Train using class defaults."""
        return cls.run_training(lr or cls.default_lr, epochs)

# Usage:
Trainer.train(lr=0.001)
# Or from CLI: python script.py --lr 0.001 --epochs 50
```

### With proto.partial

```python
class Config:
    lr: float = 0.01
    batch_size: int = 32

class Trainer:
    @proto.partial(Config)  # proto.partial on OUTSIDE
    @classmethod
    def train(cls, lr, batch_size):
        return {"lr": lr, "batch_size": batch_size}

# Config values are injected, cls is bound correctly
Trainer.train()  # → {"lr": 0.01, "batch_size": 32}
```

```{note}
The decorator order matters because Python applies decorators bottom-up.
`@proto.cli` on the outside receives the `classmethod`/`staticmethod` descriptor,
allowing it to detect and handle method binding correctly.
```

## Functions

### `proto.bind(**kwargs)`

Context manager for parameter binding.

```python
# As context manager (scoped overrides)
with proto.bind(lr=0.01, batch_size=64):
    train()

# As direct call (global overrides)
proto.bind(lr=0.01)
train()

# With prefixed parameters
proto.bind(**{"Model.name": "vit", "Training.lr": 0.0001})
```

**Parameters:**
- `**kwargs` - Parameter overrides as keyword arguments
  - Direct parameters: `lr=0.01`
  - Prefixed parameters: `**{"Model.name": "vit"}`

**Returns:**
- `BindContext` - Context manager object

**Usage:**
- Scoped overrides with `with` statement
- Global overrides with direct call
- Supports dot-notation for prefixed configs

### `proto.parse(func, **kwargs)`

Parse overrides and call a function (utility wrapper).

```python
result = proto.parse(train, lr=0.01, batch_size=64)
```

**Parameters:**
- `func` - Function to call
- `**kwargs` - Parameter overrides

**Returns:**
- Result of calling `func` with overrides applied

## Type Support

params-proto v3 supports the following type annotations:

### Basic Types
- `int` - Integer values
- `float` - Floating point values
- `str` - String values
- `bool` - Boolean values (generates `--flag` / `--no-flag`)

### Complex Types
- `Union[A, B]` or `A | B` - Union types
- `Literal["a", "b"]` - Literal values
- `Enum` - Enumeration types
- `tuple[int, ...]` - Tuple types
- `Optional[T]` or `T | None` - Optional types

### Example

```python
from typing import Literal
from enum import Enum, auto

class Optimizer(Enum):
    ADAM = auto()
    SGD = auto()

@proto
class Params:
    # Basic types
    lr: float = 0.001
    batch_size: int = 32

    # Union type
    precision: Literal["fp16", "fp32", "fp64"] = "fp32"

    # Enum
    optimizer: Optimizer = Optimizer.ADAM

    # Tuple
    image_size: tuple[int, int] = (224, 224)

    # Optional
    checkpoint: str | None = None
```

## Documentation Extraction

params-proto automatically extracts documentation from:

1. **Inline comments**: `param: type = value  # Documentation here`
2. **Function docstrings**: The main docstring becomes the program description
3. **Class docstrings**: Used for config group descriptions

### Example

```python
@proto.cli
def train(
    lr: float = 0.001,  # Learning rate - appears in help
    batch_size: int = 32,  # Batch size - appears in help
):
    """Main description - appears at top of help."""
    pass
```

Generates:
```{ansi-block}
:string_escape:

usage: train.py [-h] [--lr \x1b[1m\x1b[94mFLOAT\x1b[0m] [--batch-size \x1b[1m\x1b[94mINT\x1b[0m]

Main description - appears at top of help.

options:
  -h, --help           show this help message and exit
  --lr \x1b[1m\x1b[94mFLOAT\x1b[0m           Learning rate \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m0.001\x1b[0m\x1b[36m)\x1b[0m
  --batch-size \x1b[1m\x1b[94mINT\x1b[0m     Batch size \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m32\x1b[0m\x1b[36m)\x1b[0m
```

## Environment Variables

### `EnvVar`

Read configuration from environment variables with automatic type conversion.

**Three syntaxes:**

```python
from params_proto import proto, EnvVar

@proto.cli
def config(
    # 1. Matmul operator - simple env var
    port: int = EnvVar @ "PORT",

    # 2. Matmul + pipe - env var with fallback
    host: str = EnvVar @ "HOST" | "localhost",

    # 3. Function call - explicit syntax
    db_url: str = EnvVar("DATABASE_URL", default="sqlite:///local.db"),

    # Template expansion with $VAR or ${VAR}
    data_dir: str = EnvVar @ "$HOME/data/${PROJECT}",
):
    """Configuration from environment."""
    pass
```

**Features:**
- Automatic type conversion (str, int, float, bool)
- Template expansion with `$VAR` or `${VAR}` syntax
- Multiple variables in templates
- Fallback values with `|` operator
- Resolved at decoration time

**Type conversion:**
```python
# Environment: PORT=8080, THRESHOLD=0.75, DEBUG=true

@proto.cli
def config(
    port: int = EnvVar @ "PORT",          # → 8080 (int)
    threshold: float = EnvVar @ "THRESHOLD",  # → 0.75 (float)
    debug: bool = EnvVar @ "DEBUG",       # → True (bool)
):
    pass
```

**Template expansion:**
```python
# Environment: HOME=/home/alice, PROJECT=ml-research

@proto.cli
def paths(
    workspace: str = EnvVar @ "$HOME/projects/${PROJECT}",
    # → "/home/alice/projects/ml-research"
):
    pass
```

See **[Environment Variables Guide](../key_concepts/environment_variables.md)** for comprehensive documentation.

## Special Attributes

Decorated objects get special attributes:

### `__help_str__`

Available on `@proto.cli` decorated functions:

```python
@proto.cli
def main(lr: float = 0.001):
    """Train a model."""
    pass

print(main.__help_str__)  # Prints the generated help text
```

## Import Paths

```python
# Main import
from params_proto import proto

# All decorators are accessed via proto
@proto  # Base decorator
@proto.cli  # CLI decorator
@proto.prefix  # Prefix decorator

# Functions
proto.bind(**kwargs)  # Parameter binding
proto.parse(func, **kwargs)  # Parse and call

# Environment variables
from params_proto import EnvVar
```

## See Also

- [Configuration Basics](../key_concepts/configuration_basics.md) - Functions and classes
- [Types Guide](../key_concepts/types.md) - Supported type annotations
- [Environment Variables](../key_concepts/environment_variables.md) - EnvVar comprehensive guide
- [Quick Start](../quick_start.md) - Getting started tutorial
