# Parameter Iterators (piter)

The `piter()` function provides a lightweight, composable way to create parameter sweeps using simple dictionaries and operators. Unlike the `Sweep` class which requires `@proto` decorated classes, `piter` works with plain dictionaries and supports lazy evaluation for memory efficiency.

## Quick Start

```python
from params_proto.hyper import piter

# Create a parameter sweep from a dictionary (zips by default)
configs = piter({"lr": [0.001, 0.01], "batch_size": [32, 64]})

# Iterate over zipped configurations (2 configs)
for config in configs:
    print(config)
    # {'lr': 0.001, 'batch_size': 32}
    # {'lr': 0.01, 'batch_size': 64}

# For Cartesian product, use the * operator
configs = piter({"lr": [0.001, 0.01]}) * piter({"batch_size": [32, 64]})
# This creates 4 configs: all combinations
```

## Key Features

- **Lazy evaluation**: Configurations are generated on-the-fly, not stored in memory
- **Composable**: Combine iterators using operators (`*`, `%`, `**`)
- **Reusable**: Results are cached, so you can iterate multiple times
- **Memory efficient**: Only materializes when needed via `.to_list()` or `len()`

## Basic Usage

### Creating a piter

```python
# Lists of values are zipped element-wise (default behavior)
configs = piter({
    "lr": [0.001, 0.01, 0.1],
    "batch_size": [32, 64, 128]
})
# Produces 3 configs (zipped): (0.001, 32), (0.01, 64), (0.1, 128)

# Single values
fixed = piter({"seed": 42, "epochs": 100})
# Produces 1 config

# For Cartesian product, use * operator
configs = piter({"lr": [0.001, 0.01, 0.1]}) * piter({"batch_size": [32, 64]})
# Produces 6 configs (3 × 2)

# With prefixes for multiple parameter groups (zipped)
configs = piter({
    "model.depth": [18, 50],
    "training.lr": [0.001, 0.01]
})
# Produces 2 configs (zipped)
```

### Materializing Configs

```python
# Lazy iteration (recommended)
for config in configs:
    train(config)

# Convert to list (materializes all configs)
config_list = configs.to_list()
# or
config_list = configs.list

# Get length (materializes internally)
num_configs = len(configs)
```

## Operators

### Cartesian Product (`*`)

Combine two parameter iterators to create all possible combinations:

```python
piter1 = piter({"lr": [0.001, 0.01]})
piter2 = piter({"batch_size": [32, 64]})

combined = piter1 * piter2

list(combined)
# [
#   {'lr': 0.001, 'batch_size': 32},
#   {'lr': 0.001, 'batch_size': 64},
#   {'lr': 0.01, 'batch_size': 32},
#   {'lr': 0.01, 'batch_size': 64}
# ]
```

**Use case**: Exploring all combinations of independent hyperparameters.

### Override (`%`)

Apply fixed parameters to all configurations:

```python
# Create configs with Cartesian product
configs = piter({"lr": [0.001, 0.01]}) * piter({"batch_size": [32, 64]})

# Override with a dict
with_seed = configs % {"seed": 42, "device": "cuda"}

list(with_seed)
# [
#   {'lr': 0.001, 'batch_size': 32, 'seed': 42, 'device': 'cuda'},
#   {'lr': 0.001, 'batch_size': 64, 'seed': 42, 'device': 'cuda'},
#   {'lr': 0.01, 'batch_size': 32, 'seed': 42, 'device': 'cuda'},
#   {'lr': 0.01, 'batch_size': 64, 'seed': 42, 'device': 'cuda'}
# ]

# Override with another piter (uses first config)
with_defaults = configs % piter({"seed": 42, "device": "cuda"})
```

**Use case**: Adding fixed parameters (seed, device, logging config) to all experiments.

### Repeat (`**`)

Repeat each configuration n times:

```python
configs = piter({"lr": [0.001, 0.01]})

repeated = configs ** 3

list(repeated)
# [
#   {'lr': 0.001},
#   {'lr': 0.001},
#   {'lr': 0.001},
#   {'lr': 0.01},
#   {'lr': 0.01},
#   {'lr': 0.01}
# ]
```

**Use case**: Running multiple trials/seeds for each configuration.

## Composition Patterns

### Pattern 1: Grid Search with Fixed Seed

```python
# Grid search over hyperparameters (use * for Cartesian product)
grid = (
    piter({"lr": [0.001, 0.01, 0.1]}) *
    piter({"batch_size": [32, 64, 128]}) *
    piter({"weight_decay": [0.0, 0.0001, 0.001]})
)

# Add fixed seed to all configs
experiments = grid % {"seed": 42}

# 27 configs (3 × 3 × 3), all with seed=42
```

### Pattern 2: Multiple Trials per Config

```python
# Define hyperparameter search space (Cartesian product)
configs = piter({"lr": [0.001, 0.01]}) * piter({"batch_size": [32, 64]})

# Run 5 trials per config with different seeds
trials = configs ** 5

# 20 total runs (4 configs × 5 trials)
```

### Pattern 3: Combining Multiple Parameter Groups

```python
# Model architecture variations
models = piter({"model.type": ["resnet18", "resnet50", "vit"]})

# Training hyperparameters (use * for Cartesian product)
training = piter({"training.lr": [0.001, 0.01]}) * piter({"training.batch_size": [32, 64]})

# All combinations
experiments = models * training

# 12 configs (3 models × 2 lr × 2 batch_size)
```

### Pattern 4: Chained Composition

```python
# Build complex sweep by chaining operators
experiments = ((
    piter({"model": ["resnet", "vit"]}) *
    piter({"lr": [0.001, 0.01]}) *
    piter({"batch_size": [32, 64]})
) % {"seed": 42, "device": "cuda"}) ** 3

# 24 total runs (2 × 2 × 2 = 8 configs, 3 trials each)
```

## Integration with Sweep

The `Sweep` class also supports `piter` operators:

```python
from params_proto import proto, Sweep
from params_proto.hyper import piter

@proto
class Config:
    lr: float = 0.001
    batch_size: int = 32

# Create sweep the traditional way
sweep = Sweep(Config)
with sweep.product:
    Config.lr = [0.001, 0.01]
    Config.batch_size = [32, 64]

# Use operators on Sweep objects
with_seed = sweep % {"seed": 42}
repeated = sweep ** 3

# Can also mix Sweep with piter
combined = sweep * piter({"optimizer": ["adam", "sgd"]})
```

## Comparison: piter vs Sweep

| Feature | `piter` | `Sweep` |
|---------|---------|---------|
| **Input** | Plain dictionaries | `@proto` decorated classes |
| **Syntax** | `piter({"lr": [0.001, 0.01]})` | `with sweep.product: Config.lr = [0.001, 0.01]` |
| **Default behavior** | Zip (element-wise) | Context-dependent (`.product`, `.zip`, etc.) |
| **Operators** | `*` (product), `%` (override), `**` (repeat) | Context managers (`.product`, `.zip`, `.set`) |
| **Lazy** | Yes | No (materializes in context) |
| **Type checking** | No | Yes (via `@proto`) |
| **Proto integration** | No | Yes (updates class attributes) |
| **Use case** | Quick sweeps, scripting | Production configs, type safety |

## Best Practices

### 1. Use descriptive keys

```python
# Good: Clear parameter names with prefixes
piter({
    "model.depth": [18, 50],
    "training.lr": [0.001, 0.01],
    "training.optimizer": ["adam", "sgd"]
})

# Avoid: Ambiguous names
piter({"d": [18, 50], "l": [0.001, 0.01]})
```

### 2. Materialize only when necessary

```python
# Good: Iterate lazily
for config in experiments:
    train(config)

# Avoid: Unnecessary materialization
all_configs = experiments.to_list()  # Uses memory
for config in all_configs:
    train(config)
```

### 3. Use operators for clarity

```python
# Good: Use * for Cartesian product
grid = piter({"lr": [0.001, 0.01]}) * piter({"batch_size": [32, 64]})
# 4 configs: all combinations

# Good: Use zip (default) for related parameters
paired = piter({"lr": [0.001, 0.01], "weight_decay": [0.0001, 0.001]})
# 2 configs: (0.001, 0.0001) and (0.01, 0.001)

# Good: Use % for fixed values
with_defaults = piter({"lr": [0.001, 0.01]}) % {"seed": 42, "device": "cuda"}

# Avoid: Mixing independent parameters in single dict (implicit zip)
mixed = piter({"lr": [0.001, 0.01], "batch_size": [32, 64]})
# Only 2 configs (zipped), might not be what you want for grid search
```

### 4. Combine with type-safe configs in production

```python
from params_proto import proto
from params_proto.hyper import piter

@proto
class Config:
    lr: float = 0.001
    batch_size: int = 32
    seed: int = 42

# Use piter for sweep definition (Cartesian product for grid search)
sweep_configs = (
    piter({"lr": [0.001, 0.01, 0.1]}) *
    piter({"batch_size": [32, 64]})
) % {"seed": 42}

# Apply to typed config
for overrides in sweep_configs:
    Config._update(overrides)
    train()  # Config.lr, Config.batch_size are type-checked
```

## Advanced Examples

### Conditional Parameter Sweeps

```python
# Different learning rates for different optimizers
adam_configs = piter({"optimizer": "adam"}) * piter({"lr": [0.0001, 0.001, 0.01]})

sgd_configs = (
    piter({"optimizer": "sgd"}) *
    piter({"lr": [0.01, 0.1, 1.0]}) *
    piter({"momentum": [0.9, 0.95]})
)

# Combine into single sweep (use list concatenation)
all_configs = adam_configs.to_list() + sgd_configs.to_list()
# 3 adam configs + 6 sgd configs = 9 total
```

### Nested Grids with Fixed Outer Parameters

```python
# Coarse grid
coarse = piter({"lr": [0.001, 0.01, 0.1]})

# For each coarse lr, fine-tune batch size
fine_tuned = []
for coarse_config in coarse:
    fine = piter({"batch_size": [16, 32, 64, 128]}) % coarse_config
    fine_tuned.extend(fine.to_list())

# 12 total configs (3 lr × 4 batch_size)
```

### Hierarchical Parameter Groups

```python
# Dataset variations
datasets = piter({"data.name": ["cifar10", "cifar100", "imagenet"]})

# Model architectures per dataset
cifar_models = piter({"model.type": ["resnet18", "resnet34"]})
imagenet_models = piter({"model.type": ["resnet50", "resnet101"]})

# Training configs
training = piter({"training.lr": [0.001, 0.01]})

# Compose based on dataset
cifar10_exps = piter({"data.name": "cifar10"}) * cifar_models * training
cifar100_exps = piter({"data.name": "cifar100"}) * cifar_models * training
imagenet_exps = piter({"data.name": "imagenet"}) * imagenet_models * training

# Combine all
all_experiments = (
    cifar10_exps.to_list() +
    cifar100_exps.to_list() +
    imagenet_exps.to_list()
)
```

## API Reference

### `piter(spec: dict) -> ParameterIterator`

Create a parameter iterator from a specification dictionary.

**Args:**
- `spec`: Dict mapping parameter names (strings) to values or lists of values

**Returns:**
- `ParameterIterator` that zips parameter lists element-wise

**Note:** For Cartesian product, use the `*` operator to combine multiple `piter` instances.

### `ParameterIterator` Methods

| Method | Description |
|--------|-------------|
| `__iter__()` | Iterate over configurations |
| `to_list()` | Materialize all configs to a list |
| `.list` | Property alias for `to_list()` |
| `__len__()` | Return number of configs (materializes) |
| `__mul__(other)` | Cartesian product with another iterator (`*`) |
| `__mod__(other)` | Apply overrides to all configs (`%`) |
| `__pow__(n)` | Repeat each config n times (`**`) |

## Related

- [Core Concepts](core-concepts) - Decorators and basic usage
- [Configuration Patterns](configuration-patterns) - Function vs class configurations
- [CLI Fundamentals](cli-fundamentals) - Basic CLI features
- [Parameter Overrides](parameter-overrides) - Override methods and context managers
- [Hyperparameter Sweeps](hyperparameter-sweeps) - Traditional `Sweep` class with `@proto` integration
- [Advanced Patterns](advanced-patterns) - Prefixes and composition
