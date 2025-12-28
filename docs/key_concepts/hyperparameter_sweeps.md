# Hyperparameter Sweeps

Hyperparameter sweeps are essential for finding optimal configurations in machine learning and experimentation. This
guide shows you how to perform systematic parameter searches using params-proto. Simply wrap the configuration class
or the cli entrypoint function with the `Sweep` class:

> **Looking for a lighter alternative?** Check out [`piter`](parameter-iteration) for a dictionary-based approach with operator composition (`*`, `%`, `**`).

```python
from params_proto import proto, Sweep


@proto
class Params:    
  """Training configuration."""
  lr: float = 0.001  # Learning rate
  batch_size: int = 32  # Batch size
  
@proto.cli
def train(seed):
  print(seed);

# here we define the sweep:
sweep = Sweep(Params, train)

with sweep.zip as Params, train:
  train.seed = [100, 100, 100]
  Params.batch_size = [32, 64, 128]
  Params.lr = [0.001, 0.01, 0.1]

# This produces 3 configurations (zipped together)
assert tuple(sweep) == (
  {"seed": 100, "params.batch_size": 32, "params.lr": 0.001},
  {"seed": 100, "params.batch_size": 64, "params.lr": 0.01},
  {"seed": 100, "params.batch_size": 128, "params.lr": 0.1},
)
```

## Sweep Operators Reference

The `Sweep` class provides several operators for combining parameter configurations:

| Operator | Behavior | Example | Result |
|----------|----------|---------|--------|
| `sweep.product` | Cartesian product of all parameter lists | `lr=[0.001, 0.01]`<br>`batch=[32, 64]` | 4 configs:<br>`(0.001, 32)`, `(0.001, 64)`<br>`(0.01, 32)`, `(0.01, 64)` |
| `sweep.zip` | Zip parameter lists element-wise | `lr=[0.001, 0.01]`<br>`batch=[32, 64]` | 2 configs:<br>`(0.001, 32)`, `(0.01, 64)` |
| `sweep.chain` | Concatenate multiple sweep configurations | Two sweeps with 15 configs each | 30 configs total (concatenated) |
| `sweep.set` | Set fixed values for parameters | `seed=42` in outer scope | All configs inherit `seed=42` |
| `sweep.each(fn)` | Apply function to each config (for derived params) | `fn` computes `postfix` from `seed` | Dynamic parameter computation |

### Combining Operators

Operators can be nested to create complex sweep patterns:

```python
with Sweep(Params) as sweep:
    with sweep.set:
        Params.seed = 42  # Fixed for all

        with sweep.product:
            Params.lr = [0.001, 0.01, 0.1]

            with sweep.zip:
                Params.env = ["small", "large"]
                Params.batch = [32, 128]

# Produces 3 × 2 = 6 configs:
# lr=0.001, env="small", batch=32, seed=42
# lr=0.001, env="large", batch=128, seed=42
# lr=0.01, env="small", batch=32, seed=42
# lr=0.01, env="large", batch=128, seed=42
# lr=0.1, env="small", batch=32, seed=42
# lr=0.1, env="large", batch=128, seed=42
```

### Advanced: Dynamic Parameters with `.each()`

Use `.each()` to compute parameters that depend on other sweep values:

```python
@proto.prefix
class Config:
    seed: int = 10
    experiment_name: str = "default"

with Sweep(Config).product as sweep:
    Config.seed = [10, 20, 30]

@sweep.each
def compute_name(Config):
    Config.experiment_name = f"run-seed-{Config.seed}"

# Produces:
# {"Config.seed": 10, "Config.experiment_name": "run-seed-10"}
# {"Config.seed": 20, "Config.experiment_name": "run-seed-20"}
# {"Config.seed": 30, "Config.experiment_name": "run-seed-30"}
```

## Utility Methods

The `Sweep` class also provides utility methods for managing and persisting sweep configurations:

| Method | Description | Example Usage |
|--------|-------------|---------------|
| `sweep.save(filename, overwrite=True)` | Save sweep configs to JSONL file | `sweep.save("experiments.jsonl")` |
| `Sweep.read(filename)` | Read JSONL file into list of dicts | `configs = Sweep.read("experiments.jsonl")` |
| `sweep.load(file, strict=True)` | Load sweep from JSONL file or list | `sweep.load("experiments.jsonl")` |
| `Sweep.log(deps, filename)` | Append single config to JSONL file | `Sweep.log(config, "results.jsonl")` |
| `sweep.list` | Convert sweep to list of config dicts | `all_configs = sweep.list` |
| `sweep.dataframe` | Convert sweep to pandas DataFrame | `df = sweep.dataframe` |
| `sweep[index]` | Index or slice into sweep configs | `sweep[0]`, `sweep[:10]`, `sweep[::2]` |

### Example: Save and Load Sweeps

```python
from params_proto import proto, Sweep

@proto.prefix
class Params:
    lr: float = 0.001
    batch_size: int = 32

# Create and save a sweep
with Sweep(Params).product as sweep:
    Params.lr = [0.001, 0.01, 0.1]
    Params.batch_size = [32, 64]

sweep.save("my_sweep.jsonl")

# Later, load the sweep
sweep = Sweep(Params).load("my_sweep.jsonl")
for config in sweep:
    print(config)
```

### Example: Using DataFrame for Analysis

```python
# Convert sweep to pandas DataFrame for analysis
df = sweep.dataframe
print(df.describe())

# Filter configurations
high_lr = df[df["Params.lr"] > 0.005]
```

## Related

- [Core Concepts](core-concepts) - Decorators and basic usage
- [Configuration Patterns](configuration-patterns) - Function vs class configurations
- [Parameter Overrides](parameter-overrides) - Override methods and context managers
- [Parameter Iteration](parameter-iteration) - Lightweight sweeps with piter
- [Advanced Patterns](advanced-patterns) - Prefixes and composition
