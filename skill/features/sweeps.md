---
title: Hyperparameter Sweeps
description: Systematic hyperparameter exploration with Sweep
---

# Hyperparameter Sweeps

params-proto provides `Sweep` for systematic hyperparameter exploration.

## Basic Usage

```python
from params_proto import proto, Sweep

@proto.cli
def train(lr: float = 0.001, batch_size: int = 32):
    print(f"Training with lr={lr}, batch={batch_size}")

# Create sweep
sweep = Sweep(train)
```

## Grid Search (Product)

```python
sweep = Sweep(train).product(
    lr=[0.001, 0.01, 0.1],
    batch_size=[32, 64, 128],
)

# 3 × 3 = 9 configurations
for config in sweep:
    train(**config)
```

## Zip (Paired Values)

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

## Combined Sweeps

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

## Chain (Sequential)

```python
sweep = Sweep(train).chain(
    {"lr": 0.001, "batch_size": 32},
    {"lr": 0.01, "batch_size": 64},
    {"lr": 0.1, "batch_size": 128},
)
```

## Set (Fixed Values)

```python
sweep = Sweep(train).set(
    epochs=100,  # Fixed for all configs
).product(
    lr=[0.001, 0.01, 0.1],
)
```

## With @proto.prefix

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

## Iteration

```python
# As iterator
for config in sweep:
    train(**config)

# As list
configs = list(sweep)
print(f"Total: {len(configs)} configurations")

# With index
for i, config in enumerate(sweep):
    print(f"Config {i}: {config}")
```

## Saving and Loading

```python
# Save to file
sweep.save("sweep.yaml")

# Load from file
sweep = Sweep.load("sweep.yaml")
```

## DataFrame Export

```python
import pandas as pd

df = sweep.to_dataframe()
print(df)
```

## Operators

```python
# Multiplication = product
sweep = Sweep(train) * {"lr": [0.001, 0.01]} * {"batch_size": [32, 64]}

# Power = repeat
sweep = Sweep(train) ** 3  # 3 repetitions

# Modulo with dict = set
sweep = Sweep(train) % {"epochs": 100}
```

## Common Patterns

### Learning Rate Search

```python
sweep = Sweep(train).product(
    lr=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
)
```

### Architecture Search

```python
sweep = Sweep(train).product(
    model=["resnet18", "resnet50", "vit"],
    dropout=[0.1, 0.3, 0.5],
)
```

### Random Seeds

```python
sweep = Sweep(train).product(
    seed=[42, 123, 456, 789, 1337],
).set(
    lr=0.001,
    batch_size=32,
)
```

### Ablation Study

```python
# Baseline
baseline = {"lr": 0.001, "batch_size": 32, "augment": False}

sweep = Sweep(train).chain(
    baseline,
    {**baseline, "augment": True},
    {**baseline, "lr": 0.01},
    {**baseline, "batch_size": 64},
)
```

## Best Practices

1. **Start small** - Test with few values first
2. **Use meaningful ranges** - Based on domain knowledge
3. **Track results** - Log metrics for each config
4. **Parallelize** - Run configs in parallel when possible
5. **Save sweeps** - For reproducibility
