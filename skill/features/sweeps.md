---
title: Hyperparameter Sweeps
description: Systematic hyperparameter exploration with Sweep and piter
---

# Hyperparameter Sweeps

params-proto provides two approaches for systematic hyperparameter exploration:
- **`piter`** - Lightweight, composable parameter iterator (recommended)
- **`Sweep`** - Class-based sweeps with `@proto` integration

## piter - Parameter Iterator (Recommended)

The `piter` function creates parameter sweeps from plain dictionaries using a clean `@` syntax.

### Basic Usage

```python
from params_proto.hyper import piter

# Create a parameter sweep (zips values by default)
configs = piter @ {"lr": [0.001, 0.01], "batch_size": [32, 64]}

for config in configs:
    print(config)
    # {'lr': 0.001, 'batch_size': 32}
    # {'lr': 0.01, 'batch_size': 64}
```

### Cartesian Product with `*`

Use `*` to create all combinations. Only the first dict needs `piter @`:

```python
# Grid search: 4 configs (2 × 2)
configs = piter @ {"lr": [0.001, 0.01]} * {"batch_size": [32, 64]}

for config in configs:
    print(config)
    # {'lr': 0.001, 'batch_size': 32}
    # {'lr': 0.001, 'batch_size': 64}
    # {'lr': 0.01, 'batch_size': 32}
    # {'lr': 0.01, 'batch_size': 64}
```

### Chaining Multiple Products

```python
# 3-way product: 8 configs (2 × 2 × 2)
configs = piter @ {"lr": [0.001, 0.01]} * {"batch_size": [32, 64]} * {"model": ["resnet", "vit"]}

for config in configs:
    train(**config)
```

### Override with `%`

Apply fixed parameters to all configurations:

```python
# Add seed to all configs
configs = piter @ {"lr": [0.001, 0.01]} * {"batch_size": [32, 64]} % {"seed": 42}

# All 4 configs have seed=42
```

### Repeat with `**`

Run multiple trials per config:

```python
# 2 configs × 3 trials = 6 runs
configs = (piter @ {"lr": [0.001, 0.01]}) ** 3
```

### Complex Composition

```python
# Grid search with fixed seed and 3 trials
experiments = (
    piter @ {"lr": [0.001, 0.01, 0.1]}
    * {"batch_size": [32, 64]}
    * {"model": ["resnet", "vit"]}
) % {"seed": 42} ** 3

# 12 configs × 3 trials = 36 runs
for config in experiments:
    train(**config)
```

### With @proto.cli

```python
from params_proto import proto
from params_proto.hyper import piter

@proto.cli
def train(lr: float = 0.001, batch_size: int = 32, seed: int = 42):
    print(f"Training: lr={lr}, batch={batch_size}, seed={seed}")

# Run sweep
for config in piter @ {"lr": [0.001, 0.01]} * {"batch_size": [32, 64]}:
    train(**config)
```

---

## Sweep - Class-Based Sweeps

For integration with `@proto` decorated classes, use `Sweep`.

### Basic Usage

```python
from params_proto import proto, Sweep

@proto.cli
def train(lr: float = 0.001, batch_size: int = 32):
    print(f"Training with lr={lr}, batch={batch_size}")

# Create sweep
sweep = Sweep(train)
```

### Grid Search (Product)

```python
sweep = Sweep(train).product(
    lr=[0.001, 0.01, 0.1],
    batch_size=[32, 64, 128],
)

# 3 × 3 = 9 configurations
for config in sweep:
    train(**config)
```

### Zip (Paired Values)

```python
sweep = Sweep(train).zip(
    lr=[0.001, 0.01, 0.1],
    batch_size=[32, 64, 128],
)

# 3 configurations (paired)
# (0.001, 32), (0.01, 64), (0.1, 128)
for config in sweep:
    train(**config)
```

### Combined Sweeps

```python
sweep = Sweep(train).product(
    lr=[0.001, 0.01],
).zip(
    batch_size=[32, 64],
    epochs=[100, 200],
)

# 2 × 2 = 4 configurations
# lr varies independently, batch_size and epochs are paired
```

### Chain (Sequential)

```python
sweep = Sweep(train).chain(
    {"lr": 0.001, "batch_size": 32},
    {"lr": 0.01, "batch_size": 64},
    {"lr": 0.1, "batch_size": 128},
)
```

### Set (Fixed Values)

```python
sweep = Sweep(train).set(
    epochs=100,  # Fixed for all configs
).product(
    lr=[0.001, 0.01, 0.1],
)
```

### With @proto.prefix

```python
@proto.prefix
class Training:
    lr: float = 0.001
    batch_size: int = 32

@proto.cli
def main(seed: int = 42):
    print(f"lr={Training.lr}, batch={Training.batch_size}")

# Sweep over prefix class
sweep = Sweep(Training).product(
    lr=[0.001, 0.01],
    batch_size=[32, 64],
)

for config in sweep:
    with proto.bind(Training, **config):
        main()
```

### Sweep Operators

```python
# Multiplication = product
sweep = Sweep(train) * {"lr": [0.001, 0.01]} * {"batch_size": [32, 64]}

# Power = repeat
sweep = Sweep(train) ** 3  # 3 repetitions

# Modulo with dict = set
sweep = Sweep(train) % {"epochs": 100}
```

### Saving and Loading

```python
# Save to file
sweep.save("sweep.yaml")

# Load from file
sweep = Sweep.load("sweep.yaml")
```

---

## Comparison: piter vs Sweep

| Feature | `piter` | `Sweep` |
|---------|---------|---------|
| **Syntax** | `piter @ {"lr": [0.001, 0.01]}` | `Sweep(train).product(lr=[...])` |
| **Cartesian product** | `piter @ {...} * {...}` | `.product(...)` |
| **Input** | Plain dictionaries | `@proto` decorated classes |
| **Lazy evaluation** | Yes | No |
| **Type checking** | No | Yes (via `@proto`) |
| **Use case** | Quick sweeps, scripting | Production configs, type safety |

## Best Practices

1. **Use `piter @` for quick experiments** - Clean syntax, easy composition
2. **Use `Sweep` for production** - Type safety, proto integration
3. **Start small** - Test with few values first
4. **Use meaningful ranges** - Based on domain knowledge
5. **Track results** - Log metrics for each config
6. **Save sweeps** - For reproducibility
