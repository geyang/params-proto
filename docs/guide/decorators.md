# Decorators

params-proto v3 provides three main decorators for defining configurations:

- `@proto` - Base decorator for classes and functions
- `@proto.cli` - Decorator for CLI entry points
- `@proto.prefix` - Decorator for singleton configuration groups

## `@proto` - Base Decorator

The `@proto` decorator is the foundation of params-proto. It works with both classes and functions.

### With Classes

```python
from params_proto import proto

@proto
class Config:
    """My configuration."""
    lr: float = 0.001  # Learning rate
    batch_size: int = 32  # Batch size
    epochs: int = 100  # Number of epochs
```

The decorated class can be used normally:

```python
# Create instance
config = Config()
print(config.lr)  # 0.001

# Override via attribute
Config.lr = 0.01
config2 = Config()
print(config2.lr)  # 0.01
```

### With Functions

```python
@proto
def train(
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Batch size
):
    """Train a model."""
    print(f"Training with lr={lr}, batch_size={batch_size}")
```

Call it normally:

```python
# Use defaults
train()  # Training with lr=0.001, batch_size=32

# Override with kwargs
train(lr=0.01, batch_size=64)  # Training with lr=0.01, batch_size=64

# Override with attributes
train.lr = 0.01
train()  # Training with lr=0.01, batch_size=32
```

## `@proto.cli` - CLI Entry Points

The `@proto.cli` decorator extends `@proto` with automatic CLI parsing and help generation.

```python
@proto.cli
def main(
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Batch size
    epochs: int = 100,  # Number of epochs
):
    """Train a neural network."""
    print(f"Training: lr={lr}, batch={batch_size}, epochs={epochs}")

if __name__ == "__main__":
    main()
```

### Automatic Help Generation

Running `python script.py --help` produces:

```
usage: script.py [-h] [--lr FLOAT] [--batch-size INT] [--epochs INT]

Train a neural network.

options:
  -h, --help           show this help message and exit
  --lr FLOAT           Learning rate (default: 0.001)
  --batch-size INT     Batch size (default: 32)
  --epochs INT         Number of epochs (default: 100)
```

### CLI Usage

```bash
# Use defaults
python script.py

# Override parameters
python script.py --lr 0.01 --batch-size 64

# Boolean flags
python script.py --use-cuda  # Sets use_cuda=True
python script.py --no-use-cuda  # Sets use_cuda=False
```

### Documentation Styles

Inline comments become help text:

```python
@proto.cli
def train(
    lr: float = 0.001,  # Learning rate for the optimizer
    batch_size: int = 32,  # Number of samples per batch
):
    """Main docstring appears in help."""
    pass
```

## `@proto.prefix` - Singleton Configs

The `@proto.prefix` decorator creates singleton configuration groups with automatic CLI prefixes.

```python
@proto.prefix
class Model:
    """Model configuration."""
    name: str = "resnet50"  # Model architecture
    pretrained: bool = True  # Use pretrained weights

@proto.prefix
class Training:
    """Training hyperparameters."""
    lr: float = 0.001  # Learning rate
    batch_size: int = 32  # Batch size

@proto.cli
def main(seed: int = 42):  # Random seed
    """Train a model."""
    print(f"Training {Model.name}")
    print(f"  LR: {Training.lr}, Batch: {Training.batch_size}")
```

### Singleton Behavior

Prefixed configs are singletons - there's only one instance:

```python
print(Model.name)  # resnet50 (access directly)

Model.name = "vit"  # Set globally
print(Model.name)  # vit (changed globally)
```

### CLI with Prefixes

```bash
$ python main.py --help
usage: main.py [-h] [--seed INT] [OPTIONS]

Train a model.

options:
  -h, --help                   show this help message and exit
  --seed INT                   Random seed (default: 42)

Model options:
  Model configuration.

  --Model.name STR             Model architecture (default: resnet50)
  --Model.pretrained           Use pretrained weights (default: True)

Training options:
  Training hyperparameters.

  --Training.lr FLOAT          Learning rate (default: 0.001)
  --Training.batch-size INT    Batch size (default: 32)

$ python main.py --Model.name vit --Training.lr 0.0001
Training vit
  LR: 0.0001, Batch: 32
```

## Decorator Comparison

| Decorator | Use Case | CLI Support | Singleton | When to Use |
|-----------|----------|-------------|-----------|-------------|
| `@proto` | General configs | No | No | Libraries, shared configs |
| `@proto.cli` | Script entry points | Yes | No | Main functions, CLI tools |
| `@proto.prefix` | Config groups | Via parent CLI | Yes | Modular configs, reusable components |

## Combining Decorators

You can mix different decorators in the same project:

```python
# Singleton config groups
@proto.prefix
class Database:
    host: str = "localhost"
    port: int = 5432

@proto.prefix
class API:
    timeout: float = 30.0
    max_retries: int = 3

# Regular config (non-singleton)
@proto
class RequestConfig:
    """Per-request configuration."""
    headers: dict = {}
    timeout: float | None = None

# CLI entry point
@proto.cli
def serve(
    port: int = 8000,  # Server port
    debug: bool = False,  # Debug mode
):
    """Start the API server."""
    print(f"Server on :{port}")
    print(f"Database: {Database.host}:{Database.port}")
    print(f"API timeout: {API.timeout}s")
```

## Advanced Usage

### Decorator with Arguments

```python
# Currently, decorators don't take arguments
# Use them as shown above

@proto.cli  # Correct
def main():
    pass

# @proto.cli()  # Not supported
# def main():
#     pass
```

### Type Annotations

All decorated functions and classes require type annotations:

```python
@proto
class Config:
    # Required: type annotation
    lr: float = 0.001  # ✓ Correct

    # Not supported: no type annotation
    # batch_size = 32  # ✗ Will not work in v3
```

### Dataclasses

You can use dataclasses with `@proto`:

```python
from dataclasses import dataclass

@proto
@dataclass
class Config:
    lr: float = 0.001
    batch_size: int = 32
```

## Best Practices

### 1. Use `@proto.cli` for Entry Points

```python
# ✓ Good: Clear entry point
@proto.cli
def main(config_path: str = "config.yaml"):
    """Run the application."""
    pass

if __name__ == "__main__":
    main()
```

### 2. Use `@proto.prefix` for Modular Configs

```python
# ✓ Good: Logical grouping
@proto.prefix
class Model:
    """Model settings."""
    pass

@proto.prefix
class Training:
    """Training settings."""
    pass
```

### 3. Use `@proto` for Shared Configs

```python
# ✓ Good: Reusable config
@proto
class ExperimentConfig:
    """Config that can be instantiated multiple times."""
    pass

# Create multiple instances with different settings
exp1 = ExperimentConfig()
exp2 = ExperimentConfig()
```

### 4. Document with Inline Comments

```python
# ✓ Good: Inline documentation
@proto.cli
def train(
    lr: float = 0.001,  # Learning rate for optimizer
    batch_size: int = 32,  # Samples per training batch
):
    pass

# ✗ Avoid: No documentation
@proto.cli
def train(lr: float = 0.001, batch_size: int = 32):
    pass
```

## Next Steps

- Learn about [Functions](functions.md) for function-based configs
- Explore [Classes](classes.md) for class-based configs
- See [Types](types.md) for supported type annotations
- Check [Overrides](overrides.md) for parameter override patterns
