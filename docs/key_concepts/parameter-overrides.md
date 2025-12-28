# Parameter Overrides

params-proto v3 provides multiple ways to override parameter values at runtime, from CLI arguments to programmatic assignment.

## The Three Decorators

**Config decorators** define parameter schemas:

| Decorator | Scope | Use Case |
|-----------|-------|----------|
| `@proto` | Multiple instances | Library code, reusable components |
| `@proto.prefix` | Singleton (global) | Namespaced config groups (`Model.lr`, `Training.epochs`) |

**App decorator** creates CLI entry point:

| Decorator | Use Case |
|-----------|----------|
| `@proto.cli` | Wraps a function or class to parse CLI args |

**Typical pattern:** Define configs with `@proto`/`@proto.prefix`, then create an entry point with `@proto.cli`.

## Override Precedence

Parameters can be overridden in multiple ways, with clear precedence (highest to lowest):

1. **Function kwargs** - `train(lr=0.1)`
2. **proto.bind() context** - `proto.bind(lr=0.01)`
3. **Direct assignment** - `train.lr = 0.01`
4. **CLI arguments** - `--lr 0.01`
5. **Environment variables** - `LR=0.01`
6. **Default values** - `lr: float = 0.001`

## Override Methods

### 1. Command Line Arguments

The most common way to override parameters is through CLI arguments:

```python
from params_proto import proto

@proto.cli
def train(
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Batch size
    epochs: int = 100,  # Number of epochs
):
    """Train a model."""
    print(f"Training with lr={lr}, batch_size={batch_size}")

if __name__ == "__main__":
    train()
```

**CLI usage:**
```bash
# Override specific parameters
python train.py --lr 0.01 --batch-size 64

# Override all parameters
python train.py --lr 0.001 --batch-size 128 --epochs 200

# Boolean flags
python train.py --debug  # Set debug=True
python train.py --no-debug  # Set debug=False
```

### 2. Direct Assignment

Modify parameter values by directly assigning to the function or class:

```python
@proto.cli
def train(
    lr: float = 0.001,
    batch_size: int = 32,
):
    """Train a model."""
    print(f"lr={lr}, batch_size={batch_size}")

# Direct assignment before calling
train.lr = 0.01
train.batch_size = 64

train()  # Uses lr=0.01, batch_size=64
```

**With classes:**
```python
@proto
class Params:
    lr: float = 0.001
    batch_size: int = 32

# Modify class defaults
Params.lr = 0.01
Params.batch_size = 64

# New instances use updated defaults
config = Params()
print(config.lr)  # 0.01
```

### 3. Function kwargs

Pass parameters directly when calling the function (highest priority):

```python
@proto.cli
def train(
    lr: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
):
    """Train a model."""
    print(f"lr={lr}, batch_size={batch_size}, epochs={epochs}")

# Override via kwargs
train(lr=0.01, batch_size=64)
```

**Kwargs override all other methods:**
```python
# Set via direct assignment
train.lr = 0.001
train.batch_size = 32

# Kwargs take precedence
train(lr=0.01, batch_size=64)  # Uses 0.01 and 64, not 0.001 and 32
```

### 4. proto.bind() Context

Use `proto.bind()` to set multiple overrides at once, especially useful for testing and parameter sweeps:

```python
from params_proto import proto

@proto.prefix
class Model:
    hidden_size: int = 256
    num_layers: int = 4

@proto.prefix
class Training:
    lr: float = 0.001
    batch_size: int = 32

@proto.cli
def main(seed: int = 42):
    """Main entry point."""
    print(f"Model: hidden_size={Model.hidden_size}, num_layers={Model.num_layers}")
    print(f"Training: lr={Training.lr}, batch_size={Training.batch_size}")
    print(f"seed={seed}")

# Option 1: Function-style binding
proto.bind(
    seed=100,
    **{
        "Model.hidden_size": 512,
        "Model.num_layers": 8,
        "Training.lr": 0.01,
        "Training.batch_size": 64,
    }
)

main()  # Uses all bound values

# Option 2: Context manager (scoped)
with proto.bind(seed=200, **{"Model.hidden_size": 1024}):
    main()  # Uses seed=200, hidden_size=1024

# Values reset after context exits
main()  # Back to defaults or previous bindings
```

**Binding precedence:**
```python
# Set default
train.lr = 0.001

# Bind new value (use direct keys for @proto.cli)
proto.bind(lr=0.01)

# Kwargs still take highest priority
train(lr=0.1)  # Uses 0.1, not 0.01
```

## Override Precedence

When the same parameter is set multiple ways, params-proto uses this precedence order:

```python
@proto.cli
def train(lr: float = 0.001, batch_size: int = 32):
    """Train a model."""
    print(f"lr={lr}, batch_size={batch_size}")

# 1. Default value
# lr=0.001, batch_size=32

# 2. Environment variable (if using EnvVar)
# LR=0.01 python train.py

# 3. CLI arguments
# python train.py --lr 0.01

# 4. Direct assignment
train.lr = 0.01

# 5. proto.bind() context (direct keys for @proto.cli)
proto.bind(lr=0.01)

# 6. Function kwargs (highest priority)
train(lr=0.1)  # This wins
```

**Priority order (highest to lowest):**
1. Function kwargs
2. proto.bind() context
3. Direct assignment
4. CLI arguments
5. Environment variables
6. Default values

## Prefixed Configurations

Override prefixed configs using dotted notation:

```python
@proto.prefix
class Model:
    name: str = "resnet50"
    hidden_size: int = 256

@proto.prefix
class Training:
    lr: float = 0.001
    batch_size: int = 32

@proto.cli
def main():
    """Train model."""
    print(f"Training {Model.name} with lr={Training.lr}")

# Method 1: Direct assignment
Model.name = "vit"
Training.lr = 0.01

main()

# Method 2: CLI with prefixes (kebab-case)
# python main.py --model.name vit --training.lr 0.01

# Method 3: proto.bind() with dotted names
proto.bind(**{
    "Model.name": "vit",
    "Training.lr": 0.01,
})

main()
```

**Nested function parameters:**
```python
@proto.prefix
def train(
    lr: float = 0.001,
    batch_size: int = 32,
):
    """Training configuration."""
    return {"lr": lr, "batch_size": batch_size}

@proto.cli
def main(seed: int = 42):
    """Main function."""
    result = train()
    print(f"seed={seed}, training={result}")

# Override nested function with dict
result = main(seed=100, train={"lr": 0.01, "batch_size": 64})

# Override using proto.bind()
proto.bind(**{
    "seed": 200,
    "train.lr": 0.01,
    "train.batch_size": 64,
})
main()
```

## Common Patterns

### Testing

Override parameters in tests without modifying code:

```python
def test_training():
    """Test training with different configurations."""
    from params_proto import proto

    @proto.cli
    def train(lr: float = 0.001, epochs: int = 100):
        """Train model."""
        return {"lr": lr, "epochs": epochs}

    # Test with default values
    result = train()
    assert result["lr"] == 0.001

    # Test with overrides
    result = train(lr=0.01, epochs=10)
    assert result["lr"] == 0.01
    assert result["epochs"] == 10

    # Test with proto.bind() (direct keys for @proto.cli)
    with proto.bind(lr=0.1):
        result = train()
        assert result["lr"] == 0.1
```

### Parameter Sweeps

Use proto.bind() to run parameter sweeps:

```python
@proto.cli
def train(
    lr: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
):
    """Train model."""
    print(f"Training: lr={lr}, batch_size={batch_size}")
    # ... training code ...
    return accuracy

# Parameter sweep (direct keys for @proto.cli)
results = []
for lr in [0.001, 0.01, 0.1]:
    for batch_size in [32, 64, 128]:
        with proto.bind(lr=lr, batch_size=batch_size):
            accuracy = train()
            results.append({
                "lr": lr,
                "batch_size": batch_size,
                "accuracy": accuracy,
            })

# Find best hyperparameters
best = max(results, key=lambda x: x["accuracy"])
print(f"Best: lr={best['lr']}, batch_size={best['batch_size']}")
```

### Configuration Profiles

Create configuration profiles using proto.bind():

```python
@proto.prefix
class Model:
    hidden_size: int = 256
    num_layers: int = 4
    dropout: float = 0.1

@proto.cli
def train():
    """Train model."""
    print(f"Model config: {Model.hidden_size}, {Model.num_layers}, {Model.dropout}")

# Configuration profiles
PROFILES = {
    "small": {
        "Model.hidden_size": 128,
        "Model.num_layers": 2,
        "Model.dropout": 0.1,
    },
    "medium": {
        "Model.hidden_size": 256,
        "Model.num_layers": 4,
        "Model.dropout": 0.2,
    },
    "large": {
        "Model.hidden_size": 512,
        "Model.num_layers": 8,
        "Model.dropout": 0.3,
    },
}

# Use a profile
def train_with_profile(profile: str):
    """Train with a specific profile."""
    proto.bind(**PROFILES[profile])
    train()

train_with_profile("large")
```

### Dynamic Configuration

Build configurations dynamically:

```python
@proto.cli
def train(
    lr: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
):
    """Train model."""
    print(f"Training with lr={lr}, batch_size={batch_size}, epochs={epochs}")

def load_config(config_file: str):
    """Load configuration from file."""
    import json
    with open(config_file) as f:
        config = json.load(f)

    # Apply configuration using proto.bind() (direct keys for @proto.cli)
    proto.bind(**config)

# Load from JSON file
# config.json: {"lr": 0.01, "batch_size": 64, "epochs": 200}
load_config("config.json")
train()  # Uses values from config.json
```

### Partial Configuration

Override only some parameters:

```python
@proto.cli
def train(
    lr: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    warmup_steps: int = 1000,
):
    """Train model."""
    print(f"lr={lr}, batch_size={batch_size}, epochs={epochs}, warmup={warmup_steps}")

# Override only lr and batch_size, keep others as default
train(lr=0.01, batch_size=64)
# lr=0.01, batch_size=64, epochs=100 (default), warmup_steps=1000 (default)

# Using proto.bind() (direct keys for @proto.cli)
with proto.bind(lr=0.01):
    train()  # Only lr overridden
```

## Context Manager Usage

The proto.bind() context manager is useful for scoped overrides:

```python
@proto.cli
def train(lr: float = 0.001):
    """Train model."""
    print(f"Training with lr={lr}")

# Default behavior
train()  # lr=0.001

# Scoped override (direct keys for @proto.cli)
with proto.bind(lr=0.01):
    train()  # lr=0.01

    # Nested contexts
    with proto.bind(lr=0.1):
        train()  # lr=0.1 (inner context wins)

    train()  # lr=0.01 (back to outer context)

# Back to default after context exits
train()  # lr=0.001
```

**Multiple contexts:**
```python
@proto.prefix
class Model:
    name: str = "resnet50"

@proto.prefix
class Training:
    lr: float = 0.001

@proto.cli
def main():
    """Train model."""
    print(f"Training {Model.name} with lr={Training.lr}")

# Apply multiple overrides
with proto.bind(**{
    "Model.name": "vit",
    "Training.lr": 0.01,
}):
    main()  # Both overridden
```

## Best Practices

### 1. Use CLI for User-Facing Scripts

```python
# ✓ Good: CLI-driven configuration
@proto.cli
def train(lr: float = 0.001):
    """Train model."""
    pass

if __name__ == "__main__":
    train()  # Users override via CLI
```

### 2. Use proto.bind() for Testing

```python
# ✓ Good: Clean testing with context managers
def test_train():
    with proto.bind(lr=0.01):  # direct keys for @proto.cli
        result = train()
        assert result["lr"] == 0.01
```

### 3. Use Direct Assignment Sparingly

```python
# ✗ Avoid: Global side effects
train.lr = 0.01  # Affects all future calls

# ✓ Better: Explicit function calls
train(lr=0.01)  # Clear and local
```

### 4. Document Override Methods

```python
@proto.cli
def train(lr: float = 0.001):
    """Train a model.

    Args:
        lr: Learning rate

    CLI Usage:
        python train.py --lr 0.01

    Programmatic:
        train(lr=0.01)
        # or
        proto.bind(lr=0.01)  # direct keys for @proto.cli
        train()
    """
    pass
```

### 5. Validate Overridden Values

```python
@proto.cli
def train(
    lr: float = 0.001,
    batch_size: int = 32,
):
    """Train model."""
    # Validate overridden values
    if lr <= 0 or lr >= 1:
        raise ValueError(f"lr must be in (0, 1), got {lr}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    print(f"Training with lr={lr}, batch_size={batch_size}")
```

## Troubleshooting

### Override Not Taking Effect

**Problem:** Parameter override doesn't work

```python
@proto.cli
def train(lr: float = 0.001):
    """Train model."""
    print(f"lr={lr}")

train.lr = 0.01  # Set override
train()  # Still shows lr=0.001 ???
```

**Solution:** Check if CLI parsing is overriding your value

```python
# When called as a script with CLI args, CLI wins
# python train.py  # Uses CLI defaults, not train.lr

# To use programmatic overrides, call without CLI parsing
if __name__ == "__main__":
    # Option 1: Set before CLI parsing
    train.lr = 0.01
    train()  # Parses CLI, your override may be lost

    # Option 2: Use kwargs (highest priority)
    train(lr=0.01)  # Always works

    # Option 3: Use proto.bind() (direct keys for @proto.cli)
    proto.bind(lr=0.01)
    train()
```

### Prefix Override Not Working

**Problem:** Dotted prefix not recognized

```python
@proto.prefix
class Model:
    name: str = "resnet50"

# ✗ Wrong: No effect
proto.bind(Model_name="vit")

# ✓ Correct: Use dotted notation
proto.bind(**{"Model.name": "vit"})
```

### Context Manager State

**Problem:** Values persist after context exit

```python
# This should not happen, but check for:
# 1. Are you calling train() inside the context?
with proto.bind(**{"train.lr": 0.01}):
    pass  # Context exits

train()  # Should use default lr=0.001

# 2. If values persist, there may be a bug
# Report at https://github.com/geyang/params-proto/issues
```

## Related

- [Core Concepts](core-concepts) - Decorators and basic usage
- [CLI Fundamentals](cli-fundamentals) - Basic CLI features
- [CLI Patterns](cli-patterns) - Advanced CLI patterns
- [Configuration Patterns](configuration-patterns) - Function vs class configurations
- [Advanced Patterns](advanced-patterns) - Prefixes and Union-based subcommands
- [Environment Variables](environment-variables) - EnvVar configuration
- [Hyperparameter Sweeps](hyperparameter-sweeps) - Declarative parameter sweeps
- [Parameter Iteration](parameter-iteration) - Lightweight sweeps with operators
