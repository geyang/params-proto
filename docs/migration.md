# Migration Guide: v2 to v3

This guide will help you migrate from params-proto v2 to v3.

## What Changed?

params-proto v3 is a complete redesign with a cleaner, more Pythonic API. The key changes are:

| Aspect | v2 | v3 |
|--------|----|----|
| **API Style** | Class inheritance (`ParamsProto`) | Decorators (`@proto`) |
| **Type Hints** | Optional | Required |
| **Documentation** | Manual docstrings | Inline comments |
| **Functions** | Not supported | Full support |
| **Union Types** | Limited support | Full support |
| **Prefixes** | Manual prefix string | `@proto.prefix` decorator |

## Quick Comparison

### v2 Syntax
```python
from params_proto import ParamsProto

class Args(ParamsProto):
    """Training arguments"""
    lr = 0.001
    batch_size = 32
    epochs = 100

# CLI parsing
Args.parse_args()
```

### v3 Syntax
```python
from params_proto import proto

@proto
class Args:
    """Training arguments"""
    lr: float = 0.001  # Learning rate
    batch_size: int = 32  # Batch size
    epochs: int = 100  # Number of epochs
```

## Step-by-Step Migration

### Step 1: Replace Inheritance with Decorator

**v2:**
```python
from params_proto import ParamsProto

class Config(ParamsProto):
    lr = 0.001
    batch_size = 32
```

**v3:**
```python
from params_proto import proto

@proto
class Params:    lr: float = 0.001
    batch_size: int = 32
```

### Step 2: Add Type Hints

Type hints are required in v3. Add them to all your parameters:

**v2:**
```python
class Config(ParamsProto):
    lr = 0.001
    batch_size = 32
    model_name = "resnet50"
    use_gpu = True
```

**v3:**
```python
@proto
class Params:    lr: float = 0.001
    batch_size: int = 32
    model_name: str = "resnet50"
    use_gpu: bool = True
```

### Step 3: Convert Documentation

Inline comments automatically become help text in v3:

**v2:**
```python
class Config(ParamsProto):
    """
    Training configuration.

    Args:
        lr: Learning rate for optimizer
        batch_size: Size of training batches
    """
    lr = 0.001
    batch_size = 32
```

**v3:**
```python
@proto
class Params:    """Training configuration."""
    lr: float = 0.001  # Learning rate for optimizer
    batch_size: int = 32  # Size of training batches
```

### Step 4: Update Prefixed Configs

The prefix system is much simpler in v3:

**v2:**
```python
from params_proto import Proto

@Proto(prefix="Model")
class ModelConfig:
    name = "resnet50"
    pretrained = True

@Proto(prefix="Training")
class TrainingConfig:
    lr = 0.001
    epochs = 100
```

**v3:**
```python
from params_proto import proto

@proto.prefix
class Model:
    name: str = "resnet50"
    pretrained: bool = True

@proto.prefix
class Training:
    lr: float = 0.001
    epochs: int = 100
```

### Step 5: CLI Entry Points

**v2:**
```python
class Args(ParamsProto):
    lr = 0.001
    batch_size = 32

Args.parse_args()

def main():
    print(f"LR: {Args.lr}")

if __name__ == "__main__":
    main()
```

**v3:**
```python
@proto.cli
def main(
    lr: float = 0.001,
    batch_size: int = 32,
):
    """Main entry point."""
    print(f"LR: {lr}")

if __name__ == "__main__":
    main()
```

Or with classes:
```python
@proto
class Args:
    lr: float = 0.001
    batch_size: int = 32

# Access directly
print(f"LR: {Args.lr}")
```

### Step 6: Parameter Overrides

Override syntax is similar but more flexible:

**v2:**
```python
Args.lr = 0.01
Args.batch_size = 64
```

**v3 (same plus more options):**
```python
# Option 1: Direct assignment
Args.lr = 0.01

# Option 2: Function kwargs
main(lr=0.01, batch_size=64)

# Option 3: Context manager
with proto.bind(lr=0.01, batch_size=64):
    main()

# Option 4: Prefixed overrides
proto.bind(**{"Training.lr": 0.01, "Model.name": "vit"})
```

## Feature Mapping

### Environment Variables

**v2:**
```python
class Config(ParamsProto):
    data_path = Proto(env="DATA_PATH", default="./data")
```

**v3:**
```python
import os

@proto
class Params:    data_path: str = os.getenv("DATA_PATH", "./data")
```

### Flags (Boolean Arguments)

**v2:**
```python
from params_proto import Flag

class Config(ParamsProto):
    debug = Flag()
    use_cuda = Flag()
```

**v3:**
```python
@proto
class Params:    debug: bool = False
    use_cuda: bool = False
```

### Nested Configs

**v2:**
```python
class ModelConfig(ParamsProto):
    name = "resnet50"

class Config(ParamsProto):
    model = ModelConfig()
    lr = 0.001
```

**v3:**
```python
@proto.prefix
class Model:
    name: str = "resnet50"

@proto
class Params:    lr: float = 0.001

# Access as Model.name from CLI or code
```

## Common Patterns

### Pattern 1: ML Training Script

**v2:**
```python
class Args(ParamsProto):
    model = "resnet50"
    lr = 0.001
    epochs = 100

Args.parse_args()

def train():
    model = load_model(Args.model)
    optimizer = Adam(lr=Args.lr)
    for epoch in range(Args.epochs):
        # ...
```

**v3:**
```python
@proto.cli
def train(
    model: str = "resnet50",
    lr: float = 0.001,
    epochs: int = 100,
):
    """Train a model."""
    model_obj = load_model(model)
    optimizer = Adam(lr=lr)
    for epoch in range(epochs):
        # ...

if __name__ == "__main__":
    train()
```

### Pattern 2: Experiment Sweeps

**v2:**
```python
from params_proto.hyper import Sweep

with Sweep(Args) as sweep:
    Args.lr = [0.001, 0.01, 0.1]
    Args.batch_size = [32, 64, 128]

for config in sweep:
    train()
```

**v3:**
```python
for lr in [0.001, 0.01, 0.1]:
    for batch_size in [32, 64, 128]:
        with proto.bind(lr=lr, batch_size=batch_size):
            train()
```

## Breaking Changes

### 1. No More `parse_args()`
v2 required calling `Args.parse_args()`. v3 handles this automatically with `@proto.cli`.

### 2. Type Hints Required
All parameters must have type annotations in v3.

### 3. Different Import Path
```python
# v2
from params_proto import ParamsProto, Proto, Flag

# v3
from params_proto import proto
```

### 4. Prefix Syntax Changed
```python
# v2
@Proto(prefix="Model")
class ModelConfig:
    pass

# v3
@proto.prefix
class Model:
    pass
```

## Gradual Migration Strategy

You can migrate gradually:

1. **Add v3 alongside v2**: Keep v2 code, add new configs with v3
2. **Migrate module by module**: Convert one module at a time
3. **Update tests**: Ensure tests pass after each module migration
4. **Remove v2 dependencies**: Once fully migrated, remove v2 imports

## Need Help?

- Check the [User Guide](key_concepts/decorators.md) for detailed v3 documentation
- See [Examples](examples/) for real-world v3 patterns
- Open an issue on [GitHub](https://github.com/geyang/params-proto/issues)

## v2 Documentation

The v2 documentation is archived at [params-proto-v2](https://github.com/geyang/params-proto-v2).
