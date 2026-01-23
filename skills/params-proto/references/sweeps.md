# Hyperparameter Sweeps Reference

## Table of Contents

- [piter - Parameter Iterator (Recommended)](#piter---parameter-iterator-recommended)
- [Sweep - Class-Based Sweeps](#sweep---class-based-sweeps)
- [Comparison: piter vs Sweep](#comparison-piter-vs-sweep)

---

## piter - Parameter Iterator (Recommended)

The `piter` function creates parameter sweeps from plain dictionaries using a clean `@` syntax.

### Basic Usage (Zip by Default)

```python
from params_proto.hyper import piter

# Zips values element-wise (default behavior)
configs = piter @ {"lr": [0.001, 0.01], "batch_size": [32, 64]}

for config in configs:
    print(config)
    # {'lr': 0.001, 'batch_size': 32}
    # {'lr': 0.01, 'batch_size': 64}
```

### Cartesian Product with `*`

Use `*` to create all combinations. Only the first dict needs `piter @`:

```python
# Grid search: 4 configs (2 x 2)
configs = piter @ {"lr": [0.001, 0.01]} * {"batch_size": [32, 64]}

# Output:
# {'lr': 0.001, 'batch_size': 32}
# {'lr': 0.001, 'batch_size': 64}
# {'lr': 0.01, 'batch_size': 32}
# {'lr': 0.01, 'batch_size': 64}
```

### Chaining Multiple Products

```python
# 3-way product: 8 configs (2 x 2 x 2)
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

Can also use another piter:

```python
configs = piter @ {"batch_size": [32, 64]} % (piter @ {"lr": 0.001, "seed": 200})
```

### Repeat with `**`

Run multiple trials per config:

```python
# 2 configs x 3 trials = 6 runs
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

# 12 configs x 3 trials = 36 runs
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

### With Prefixed Parameter Names

```python
configs = piter @ {"model.depth": [18, 50], "training.lr": [0.001, 0.01]}
# Keys match --model.depth and --training.lr CLI syntax
```

### Methods

```python
configs = piter @ {"lr": [0.001, 0.01]}

# Convert to list
config_list = configs.to_list()  # or list(configs)

# Get length
len(configs)  # 2

# Iterate multiple times (results are cached)
for c in configs: ...
for c in configs: ...  # Works again
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

sweep = Sweep(train)
```

### Grid Search (Product)

```python
sweep = Sweep(train).product(
    lr=[0.001, 0.01, 0.1],
    batch_size=[32, 64, 128],
)

# 3 x 3 = 9 configurations
for config in sweep:
    train(**config)
```

### Context Manager Syntax

```python
@proto.prefix
class Config:
    lr: float = 0.001
    batch_size: int = 32

with Sweep(Config).product as sweep:
    Config.lr = [0.001, 0.01, 0.1]
    Config.batch_size = [32, 64]

for config in sweep:
    # Config values are set automatically
    train()
```

### Zip (Paired Values)

```python
with Sweep(Config).zip as sweep:
    Config.lr = [0.001, 0.01, 0.1]
    Config.batch_size = [32, 64, 128]

# 3 configurations (paired)
# (0.001, 32), (0.01, 64), (0.1, 128)
```

### Nested Product and Zip

```python
with Sweep(Config) as sweep:
    with sweep.product:
        Config.lr = [0.001, 0.01]
        Config.epochs = [100, 200]

        with sweep.zip:
            Config.batch_size = [32, 64]
            Config.workers = [4, 8]

# 2 x 2 x 2 = 8 configs (lr x epochs x zipped(batch_size, workers))
```

### Set (Fixed Values)

```python
with Sweep(Config) as sweep:
    Config.seed = 42  # Fixed for all configs

    with sweep.product:
        Config.lr = [0.001, 0.01]
```

### Chain (Sequential)

```python
with Sweep(Config) as sweep:
    with sweep.chain:
        with sweep.set:
            Config.level = "easy"
            with sweep.product:
                Config.seed = range(5)

        with sweep.set:
            Config.level = "hard"
            with sweep.product:
                Config.seed = range(5)

# 10 configs: 5 easy + 5 hard
```

### Each (Computed Parameters)

```python
with Sweep(Config).product as sweep:
    Config.seed = [10, 20, 30]

@sweep.each
def each(Config):
    Config.exp_name = f"seed-{Config.seed}"

# Each config gets computed exp_name
```

### Sweep Operators

```python
# Multiplication = product
result = sweep * (piter @ {"extra": [1, 2]})

# Power = repeat
result = sweep ** 3  # 3 repetitions

# Modulo = override
result = sweep % {"fixed_param": 42}
```

### Saving and Loading

```python
# Save to file
sweep.save("sweep.jsonl", verbose=False)

# Load from file
sweep = Sweep(Config).load("sweep.jsonl")
```

### Slicing

```python
# Get subset of configs
subset = list(sweep[:5])
subset = list(sweep[10:20:2])
subset = list(sweep[-10:])
```

### DataFrame Conversion

```python
df = sweep.dataframe
# Pandas DataFrame with config columns
```

---

## Comparison: piter vs Sweep

| Feature | `piter` | `Sweep` |
|---------|---------|---------|
| **Syntax** | `piter @ {"lr": [...]}` | `Sweep(Config).product(...)` |
| **Cartesian product** | `piter @ {...} * {...}` | `.product(...)` |
| **Input** | Plain dictionaries | `@proto` decorated classes |
| **Type checking** | No | Yes (via `@proto`) |
| **Save/Load** | Manual | Built-in |
| **Use case** | Quick sweeps, scripting | Production configs |

### When to Use

**Use `piter @` for:**
- Quick experiments
- Scripting and notebooks
- Simple parameter grids

**Use `Sweep` for:**
- Production pipelines
- Type safety requirements
- Complex nested sweeps
- Saving/loading configurations
