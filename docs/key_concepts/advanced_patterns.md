# Advanced Patterns

This guide covers advanced configuration patterns: prefixed configurations for multi-namespace composition and Union-based subcommands.

## Prefixed Configurations

`@proto.prefix` creates singleton configuration groups with automatic CLI prefixes, enabling modular configuration composition.

### Basic Usage

Create global configuration namespaces:

```python
from params_proto import proto

@proto.prefix
class Model:
    """Model configuration."""
    name: str = "resnet50"
    hidden_size: int = 256

@proto.prefix
class Training:
    """Training hyperparameters."""
    lr: float = 0.001
    batch_size: int = 32

@proto.cli
def main(seed: int = 42):
    """Train model."""
    print(f"Training {Model.name} with lr={Training.lr}")

if __name__ == "__main__":
    main()
```

**CLI usage:**
```bash
$ python main.py --model.name vit --training.lr 0.01
Training vit with lr=0.01
```

### Singleton Behavior

Prefixed configs are singletons - one global instance per class:

```python
@proto.prefix
class Database:
    host: str = "localhost"
    port: int = 5432

# Access anywhere in code
def connect_db():
    print(f"Connecting to {Database.host}:{Database.port}")

def query_db():
    print(f"Using port: {Database.port}")

# Both functions see the same config
connect_db()  # Connecting to localhost:5432
query_db()    # Using port: 5432

# Change once, affects everywhere
Database.host = "prod.db.com"
connect_db()  # Connecting to prod.db.com:5432
```

### Multiple Prefix Groups

Organize related settings into logical modules:

```python
@proto.prefix
class Model:
    """Model architecture."""
    name: str = "resnet50"
    hidden_size: int = 256
    num_layers: int = 4

@proto.prefix
class Optimizer:
    """Optimizer configuration."""
    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0001

@proto.prefix
class Data:
    """Data loading."""
    batch_size: int = 32
    num_workers: int = 4

@proto.cli
def train(seed: int = 42, epochs: int = 100):
    """Train model."""
    print(f"Model: {Model.name} ({Model.hidden_size} hidden)")
    print(f"Optimizer: {Optimizer.name} (lr={Optimizer.lr})")
    print(f"Data: batch_size={Data.batch_size}")
```

**CLI:**
```bash
$ python train.py --model.name vit --optimizer.lr 0.01 --data.batch-size 64
Model: vit (256 hidden)
Optimizer: adam (lr=0.01)
Data: batch_size=64
```

### Real-World Example: ML Training

```python
@proto.prefix
class Model:
    """Neural network architecture."""
    name: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 1000

@proto.prefix
class Dataset:
    """Dataset configuration."""
    name: str = "imagenet"
    data_dir: str = "./data"
    num_workers: int = 4

@proto.prefix
class Training:
    """Training hyperparameters."""
    batch_size: int = 32
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0001

@proto.prefix
class Logging:
    """Logging and checkpointing."""
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 5

@proto.cli
def train(seed: int = 42, resume: str | None = None):
    """Train image classification model."""
    print(f"Training {Model.name} on {Dataset.name}")
    print(f"Batch size: {Training.batch_size}, LR: {Training.lr}")
    print(f"Logging to: {Logging.log_dir}")
```

**Usage:**
```bash
# Quick experiment
python train.py --model.name resnet18 --training.epochs 50

# Production run
python train.py \
  --model.name resnet50 --model.pretrained \
  --dataset.data-dir /mnt/data/imagenet \
  --training.batch-size 256 --training.lr 0.1 --training.epochs 300 \
  --logging.log-dir /mnt/logs/exp001
```

### When to Use Prefixes

✅ **Use `@proto.prefix` for:**
- Global configuration needed across modules
- Organizing complex systems into logical groups
- Reusable configuration components
- Multi-namespace CLI organization

❌ **Don't use for:**
- Multiple instances needed (use `@proto`)
- Simple function-local config (use parameters)

### Overriding Prefixed Configs

**Direct assignment:**
```python
Model.name = "vit"
Training.lr = 0.01
```

**CLI override:**
```bash
python main.py --model.name vit --training.lr 0.01
```

**proto.bind():**
```python
proto.bind(**{
    "Model.name": "vit",
    "Training.lr": 0.01,
})
```

See [Parameter Overrides](overrides) for details.

## Union-Based Subcommands

Use Union types with required parameters to create subcommand-like behavior.

### Key Principle

**For required parameters (no default), params-proto calls the type hint as a constructor.**

This enables Union types to act like subcommands:

```python
from dataclasses import dataclass

@dataclass
class Train:
    """Train a model."""
    lr: float = 0.001
    batch_size: int = 32
    epochs: int = 100

@dataclass
class Evaluate:
    """Evaluate a model."""
    model: str  # Required!
    batch_size: int = 64

@dataclass
class Export:
    """Export a model."""
    format: str = "onnx"
    output: str = "model.onnx"

@proto.cli
def tool(
    command: Train | Evaluate | Export,  # Required - user must choose
    verbose: bool = False,
):
    """Multi-command tool."""
    if isinstance(command, Train):
        print(f"Training: lr={command.lr}, batch_size={command.batch_size}")
    elif isinstance(command, Evaluate):
        print(f"Evaluating: {command.model}")
    elif isinstance(command, Export):
        print(f"Exporting to {command.format}: {command.output}")

if __name__ == "__main__":
    tool()
```

**CLI usage:**
```bash
# Train subcommand
python tool.py train --lr 0.01 --batch-size 64

# Evaluate subcommand
python tool.py evaluate --model checkpoint.pt

# Export subcommand
python tool.py export --format onnx

# Get help for specific subcommand
python tool.py train --help
```

### How It Works

1. **Required parameter** - `command` has no default value
2. **Union type** - `Train | Evaluate | Export` provides options
3. **Callable types** - Each type is a dataclass (callable)
4. **Automatic instantiation** - params-proto calls the selected type
5. **Isolated parameters** - Each dataclass has its own parameters

### Class Name Conversion

**Class names convert to kebab-case CLI commands:**

```python
@dataclass
class Train:      # Python: PascalCase
    pass

# CLI command: train (kebab-case)
$ python tool.py train --lr 0.01
```

**Best practices:**

```python
# ✓ Good: Simple single-word names
class Train:      # → train
class Evaluate:   # → evaluate
class Export:     # → export

# ✓ Good: Acronyms now convert properly
class HTTPServer:    # → http-server
class MLModel:       # → ml-model
class DataLoader:    # → data-loader

# ✓ Also good: Simple alternatives
class Server:     # → server
class Model:      # → model
```

See [CLI Generation](cli_generation.md#class-names-pascalcase--lowercase) for details.

### Advantages

✅ **Type-safe** - Full IDE autocomplete and type checking
✅ **No new syntax** - Uses existing Union mechanism
✅ **Composable** - Works with any callable (dataclass, class)
✅ **Isolated parameters** - Each command has its own namespace
✅ **Automatic help** - Command-specific help generation
✅ **Shared parameters** - Main function params apply to all commands

### Limitations

❌ **Not traditional subcommands** - Uses required parameters, not argparse subparsers
❌ **Single level** - Nested subcommands (like `git remote add`) not directly supported
❌ **Dispatch required** - Must handle dispatch with isinstance() checks

### Pattern: Shared Parameters

Main function parameters apply to all subcommands:

```python
@proto.cli
def tool(
    command: Train | Evaluate,  # Subcommand
    verbose: bool = False,       # Shared across all commands
    debug: bool = False,         # Shared across all commands
):
    """Tool with shared options."""
    if verbose:
        print(f"Running {command.__class__.__name__}")

    # Dispatch...
```

**CLI:**
```bash
python tool.py train --lr 0.01 --verbose
python tool.py evaluate --model pt --verbose
```

Both commands can use `--verbose`.

### Pattern: Configuration-Based Dispatch

Use configuration classes with Union types:

```python
@dataclass
class TrainConfig:
    lr: float = 0.001
    epochs: int = 100

@dataclass
class EvalConfig:
    model: str
    num_samples: int = 1000

@proto.cli
def main(config: TrainConfig | EvalConfig):
    """Main entry point."""
    if isinstance(config, TrainConfig):
        train(config)
    elif isinstance(config, EvalConfig):
        evaluate(config)

def train(config: TrainConfig):
    print(f"Training: lr={config.lr}, epochs={config.epochs}")

def evaluate(config: EvalConfig):
    print(f"Evaluating: {config.model}, samples={config.num_samples}")
```

Clean separation of command logic and configuration.

## Combining Prefixes and Union Types

Use both together for maximum flexibility:

```python
# Global config via prefixes
@proto.prefix
class Environment:
    domain: str = "cartpole"
    task: str = "swingup"

# Command-specific configs
@dataclass
class Train:
    lr: float = 0.001
    epochs: int = 100

@dataclass
class Evaluate:
    num_episodes: int = 10

# CLI entry point
@proto.cli
def main(
    command: Train | Evaluate,  # Union-based subcommands
    seed: int = 42,
):
    """Train or evaluate RL agent."""
    print(f"Environment: {Environment.domain}-{Environment.task}")

    if isinstance(command, Train):
        print(f"Training: lr={command.lr}, epochs={command.epochs}")
    elif isinstance(command, Evaluate):
        print(f"Evaluating: {command.num_episodes} episodes")
```

**CLI:**
```bash
# Override environment + train
python main.py train --Environment.domain walker --lr 0.01

# Override environment + evaluate
python main.py evaluate --Environment.task balance --num-episodes 20
```

## Configuration Profiles

Create switchable profiles using `proto.bind()`:

```python
@proto.prefix
class Model:
    hidden_size: int = 256
    num_layers: int = 4

# Define profiles
PROFILES = {
    "small": {
        "Model.hidden_size": 128,
        "Model.num_layers": 2,
    },
    "large": {
        "Model.hidden_size": 512,
        "Model.num_layers": 8,
    },
}

def apply_profile(profile: str):
    """Apply a configuration profile."""
    proto.bind(**PROFILES[profile])

# Usage
apply_profile("large")
print(Model.hidden_size)  # 512
```

## Best Practices

### 1. Use Descriptive Names

```python
# ✓ Good
@proto.prefix
class ModelArchitecture:
    hidden_size: int = 256

# ✗ Avoid
@proto.prefix
class Params:    h: int = 256
```

### 2. Group Related Settings

```python
# ✓ Good: Logical grouping
@proto.prefix
class Optimizer:
    """All optimizer settings."""
    name: str = "adam"
    lr: float = 0.001

# ✗ Avoid: Mixing unrelated
@proto.prefix
class Settings:
    optimizer: str = "adam"
    log_dir: str = "./logs"  # Unrelated
```

### 3. Document Each Group

```python
@proto.prefix
class Training:
    """Training hyperparameters and settings.

    Controls the training loop including learning rate,
    batch size, and optimization parameters.
    """
    lr: float = 0.001  # Learning rate
    batch_size: int = 32  # Batch size
```

### 4. Keep Hierarchies Flat

```python
# ✓ Good: Flat and clear
@proto.prefix
class Model:
    pass

@proto.prefix
class Training:
    pass

# ✗ Avoid: Too nested
@proto.prefix
class SystemModelArchitecture:
    pass
```

## Related

- [Core Concepts](core-concepts) - Three main decorators
- [Configuration Patterns](configuration-patterns) - Function vs class configurations
- [CLI Fundamentals](cli-fundamentals) - Basic CLI features
- [CLI Patterns](cli-patterns) - Advanced CLI patterns
- [Naming Conventions](naming-conventions) - Name conversion rules
- [Type System](type-system) - Type annotation reference
- [Parameter Overrides](parameter-overrides) - Override methods
