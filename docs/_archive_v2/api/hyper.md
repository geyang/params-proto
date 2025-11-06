# hyper module

The `hyper` module provides powerful hyperparameter search and sweep functionality for params-proto configurations.

## Core Classes

### Sweep

The main class for creating parameter sweeps and hyperparameter searches.

```{eval-rst}
.. autoclass:: params_proto.v2.hyper.Sweep
   :members:
   :undoc-members:
   :show-inheritance:
```

#### Key Features

- **Parameter Combinations**: Generate all combinations of parameter values
- **Custom Functions**: Apply functions to each parameter combination
- **Slicing Support**: Access specific subsets of the sweep
- **Save/Load**: Persist sweep configurations to files

#### Usage Example

```python
from params_proto.v2.proto import ParamsProto, Proto
from params_proto.v2.hyper import Sweep

class Config(ParamsProto):
    learning_rate = Proto("Learning rate", default=0.001)
    batch_size = Proto("Batch size", default=32)
    model = Proto("Model type", default="resnet")

# Create a sweep
sweep = Sweep(Config).product([
    Config.learning_rate << [0.001, 0.01, 0.1],
    Config.batch_size << [16, 32, 64],
    Config.model << ["resnet", "transformer"]
])

# Iterate through combinations
for config in sweep:
    print(f"lr={Config.learning_rate}, bs={Config.batch_size}, model={Config.model}")
    # Run your experiment here

# Save sweep to file
sweep.save("experiments.jsonl")
```

### Parameter Search Methods

#### Product Sweep

Generate cartesian product of all parameter combinations:

```python
sweep = Sweep(Config).product([
    Config.learning_rate << [0.001, 0.01],
    Config.epochs << [10, 20, 50]
])
# Results in 2×3 = 6 combinations
```

#### Zip Sweep

Combine parameters element-wise:

```python
sweep = Sweep(Config).zip([
    Config.learning_rate << [0.001, 0.01, 0.1],
    Config.epochs << [10, 20, 30]
])
# Results in 3 combinations: (0.001,10), (0.01,20), (0.1,30)
```

#### Chain Sweep

Concatenate multiple sweeps:

```python
sweep1 = Sweep(Config).product([Config.model << ["resnet"]])
sweep2 = Sweep(Config).product([Config.model << ["transformer"]])
combined = sweep1.chain(sweep2)
```

## Advanced Features

### Custom Processing

Apply custom functions to each sweep iteration:

```python
def run_experiment(config):
    print(f"Running with lr={config.learning_rate}")
    # Your training code here
    return {"accuracy": 0.95, "loss": 0.1}

results = []
sweep = Sweep(Config).product([
    Config.learning_rate << [0.001, 0.01, 0.1]
]).each(run_experiment)

for result in sweep:
    results.append(result)
```

### Sweep Slicing

Access specific parts of a sweep:

```python
sweep = Sweep(Config).product([
    Config.learning_rate << [0.001, 0.01, 0.1, 0.5],
    Config.batch_size << [16, 32]
])

# Get first 3 combinations
first_three = sweep[:3]

# Get every other combination
every_other = sweep[::2]

# Get specific combination
specific = sweep[5]
```

### Saving and Loading

Persist sweep configurations:

```python
# Save sweep to JSONL file
sweep.save("hyperparameter_sweep.jsonl")

# Each line contains one parameter combination
# {"Config.learning_rate": 0.001, "Config.batch_size": 32}
# {"Config.learning_rate": 0.01, "Config.batch_size": 32}
# ...
```

## Hyperparameter Search Patterns

### Grid Search

```python
from params_proto.v2.hyper import Sweep

class ModelConfig(ParamsProto):
    learning_rate = Proto(default=0.001)
    weight_decay = Proto(default=1e-4)
    dropout = Proto(default=0.1)

# Grid search over hyperparameters
sweep = Sweep(ModelConfig).product([
    ModelConfig.learning_rate << [1e-4, 1e-3, 1e-2],
    ModelConfig.weight_decay << [1e-5, 1e-4, 1e-3],
    ModelConfig.dropout << [0.0, 0.1, 0.2]
])

print(f"Total combinations: {len(sweep)}")  # 27 combinations

# Run experiments
best_accuracy = 0
best_config = None

for i, config in enumerate(sweep):
    print(f"Experiment {i+1}/{len(sweep)}")
    
    # Your training code here
    accuracy = train_model(config)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_config = config.__dict__.copy()

print(f"Best accuracy: {best_accuracy}")
print(f"Best config: {best_config}")
```

### Random Search

```python
import random

# Generate random parameter combinations
def random_search(n_trials=50):
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    weight_decays = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    dropouts = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    configs = []
    for _ in range(n_trials):
        config = {
            'learning_rate': random.choice(learning_rates),
            'weight_decay': random.choice(weight_decays),
            'dropout': random.choice(dropouts)
        }
        configs.append(config)
    
    return configs

# Use with Sweep
random_configs = random_search(20)
for config in random_configs:
    ModelConfig._update(config)
    # Run experiment
    train_model()
```

## Integration with ML Frameworks

### PyTorch Integration

```python
import torch
import torch.nn as nn
from params_proto.v2.hyper import Sweep

class TrainingConfig(ParamsProto):
    learning_rate = Proto(default=0.001)
    batch_size = Proto(default=32)
    optimizer = Proto(default="adam")

def train_pytorch_model():
    # Create model
    model = nn.Linear(10, 1)
    
    # Create optimizer based on config
    if TrainingConfig.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=TrainingConfig.learning_rate)
    elif TrainingConfig.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=TrainingConfig.learning_rate)
    
    # Training loop
    for epoch in range(10):
        # Your training code here
        pass
    
    return model

# Hyperparameter sweep
sweep = Sweep(TrainingConfig).product([
    TrainingConfig.learning_rate << [0.001, 0.01, 0.1],
    TrainingConfig.optimizer << ["adam", "sgd"]
])

for config in sweep:
    model = train_pytorch_model()
    # Evaluate and save results
```

## Complete Module Reference

```{eval-rst}
.. automodule:: params_proto.v2.hyper
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Sweep, contextmanager, defaultdict, namedtuple, ParamsProto, Meta, Proto
   :imported-members: False
```