---
title: Common Patterns
description: Real-world examples and patterns for params-proto
---

# Common Patterns

## Simple Training Script

```python
from params_proto import proto

@proto.cli
def train(
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Batch size
    epochs: int = 100,  # Training epochs
    seed: int = 42,  # Random seed
):
    """Train a model."""
    print(f"Training for {epochs} epochs with lr={lr}")

if __name__ == "__main__":
    train()
```

## Multi-Namespace ML Config

```python
from params_proto import proto

@proto.prefix
class Data:
    """Data configuration."""
    path: str = "./data"  # Data directory
    batch_size: int = 32  # Batch size
    workers: int = 4  # Data loader workers

@proto.prefix
class Model:
    """Model configuration."""
    name: str = "resnet50"  # Architecture
    pretrained: bool = True  # Use pretrained weights
    dropout: float = 0.5  # Dropout rate

@proto.prefix
class Training:
    """Training hyperparameters."""
    lr: float = 0.001  # Learning rate
    epochs: int = 100  # Number of epochs
    weight_decay: float = 1e-4  # L2 regularization

@proto.cli
def main(
    seed: int = 42,  # Random seed
    device: str = "cuda",  # Device (cuda/cpu)
):
    """Train image classifier."""
    print(f"Model: {Model.name}")
    print(f"Data: {Data.path}")
    print(f"Training: lr={Training.lr}, epochs={Training.epochs}")

if __name__ == "__main__":
    main()
```

## Environment-Based Config

```python
from params_proto import proto, EnvVar

@proto.prefix
class Database:
    host: str = EnvVar @ "DB_HOST" | "localhost"
    port: int = EnvVar @ "DB_PORT" | 5432
    user: str = EnvVar @ "DB_USER" | "postgres"
    password: str = EnvVar @ "DB_PASSWORD"
    name: str = EnvVar @ "DB_NAME" | "myapp"

@proto.prefix
class API:
    key: str = EnvVar @ "API_KEY"
    base_url: str = EnvVar @ "API_URL" | "https://api.example.com"

@proto.cli
def serve(
    port: int = EnvVar @ "PORT" | 8080,
    debug: bool = EnvVar @ "DEBUG" | False,
):
    """Start the server."""
    print(f"Connecting to {Database.host}:{Database.port}")
    print(f"Serving on port {port}")
```

## Union Types (Subcommand-like)

```python
from dataclasses import dataclass
from params_proto import proto

@dataclass
class Train:
    """Training mode."""
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 32

@dataclass
class Evaluate:
    """Evaluation mode."""
    checkpoint: str = "model.pt"
    batch_size: int = 64

@dataclass
class Export:
    """Export mode."""
    checkpoint: str = "model.pt"
    format: str = "onnx"

@proto.cli
def main(mode: Train | Evaluate | Export):
    """ML pipeline with different modes."""
    if isinstance(mode, Train):
        print(f"Training: lr={mode.lr}, epochs={mode.epochs}")
    elif isinstance(mode, Evaluate):
        print(f"Evaluating: {mode.checkpoint}")
    elif isinstance(mode, Export):
        print(f"Exporting to {mode.format}")

if __name__ == "__main__":
    main()
```

```bash
python main.py train --lr 0.01 --epochs 50
python main.py evaluate --checkpoint best.pt
python main.py export --format torchscript
```

## Hyperparameter Sweep

```python
from params_proto import proto, Sweep

@proto.cli
def train(
    lr: float = 0.001,
    batch_size: int = 32,
    model: str = "resnet50",
    seed: int = 42,
):
    """Train with given hyperparameters."""
    print(f"Training {model} with lr={lr}, batch={batch_size}, seed={seed}")
    # Return metrics for logging
    return {"accuracy": 0.95, "loss": 0.1}

# Run sweep
sweep = Sweep(train).product(
    lr=[0.001, 0.01],
    batch_size=[32, 64],
    model=["resnet50", "vit"],
).set(
    seed=42,
)

results = []
for config in sweep:
    metrics = train(**config)
    results.append({**config, **metrics})
```

## Context Manager Overrides

```python
from params_proto import proto

@proto.prefix
class Config:
    lr: float = 0.001
    debug: bool = False

def train():
    print(f"lr={Config.lr}, debug={Config.debug}")

# Default values
train()  # lr=0.001, debug=False

# Override with context manager
with proto.bind(Config, lr=0.01, debug=True):
    train()  # lr=0.01, debug=True

# Back to defaults
train()  # lr=0.001, debug=False
```

## Reusable Config Class

```python
from params_proto import proto

@proto
class OptimizerConfig:
    """Reusable optimizer configuration."""
    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 1e-4

# Create multiple instances
adam_config = OptimizerConfig(name="adam", lr=0.001)
sgd_config = OptimizerConfig(name="sgd", lr=0.01)

# Use in training
def train(optimizer_config: OptimizerConfig):
    print(f"Using {optimizer_config.name} with lr={optimizer_config.lr}")
```

## CLI with Validation

```python
from params_proto import proto
from enum import Enum

class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

@proto.cli
def train(
    lr: float = 0.001,  # Learning rate (0, 1)
    batch_size: int = 32,  # Batch size (power of 2)
    precision: Precision = Precision.FP32,  # Training precision
):
    """Train with validation."""
    # Validate
    if not 0 < lr < 1:
        raise ValueError(f"lr must be in (0, 1), got {lr}")
    if batch_size & (batch_size - 1) != 0:
        raise ValueError(f"batch_size must be power of 2, got {batch_size}")

    print(f"Training with lr={lr}, batch={batch_size}, precision={precision.value}")

if __name__ == "__main__":
    train()
```

## Testing Pattern

```python
from params_proto import proto

@proto.cli(prog="train")  # Fixed name for reproducible help text
def train(lr: float = 0.001):
    """Train a model."""
    return {"lr": lr}

# Test programmatically (bypasses CLI parsing)
def test_train():
    result = train(lr=0.01)
    assert result["lr"] == 0.01

# Test help text
def test_help():
    assert "--lr FLOAT" in train.__help_str__
    assert "(default: 0.001)" in train.__help_str__
```
