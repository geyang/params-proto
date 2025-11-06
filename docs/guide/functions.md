# Function CLI with Type Hints

params-proto v3 uses Python type hints and inline comments to automatically generate CLI help documentation.

## Basic Function CLI

Convert any function into a CLI program with `@proto.cli`:

```python
from params_proto import proto


@proto.cli
def train_mnist(
    batch_size: int = 128,  # Training batch size
    learning_rate: float = 0.001,  # Learning rate
    epochs: int = 10,  # Number of epochs
):
    """Train an MLP on MNIST dataset."""
    print(f"Training with batch_size={batch_size}")


if __name__ == "__main__":
    train_mnist()
```

Running `python train_mnist.py --help`:

```
usage: train_mnist.py [-h] [--batch-size INT] [--learning-rate FLOAT] [--epochs INT]

Train an MLP on MNIST dataset.

options:
  -h, --help           show this help message and exit
  --batch-size INT     Training batch size (default: 128)
  --learning-rate FLOAT
                       Learning rate (default: 0.001)
  --epochs INT         Number of epochs (default: 10)
```

## Documentation Syntaxes

params-proto supports multiple ways to document parameters, listed by priority:

### 1. Inline Comments (Recommended)

Use `#` comments on the same line as the parameter:

```python
@proto.cli
def train(
    batch_size: int = 128,  # Training batch size
    learning_rate: float = 0.001,  # Initial learning rate
):
    """Train a model."""
    pass
```

**Pros:**
- Clean and concise
- Documentation right next to the parameter
- IDE-friendly
- Works with all parameter styles

**Cons:**
- Limited to single line

### 2. Args Section in Docstring

Use Google-style or NumPy-style docstrings:

```python
@proto.cli
def train(
    batch_size: int = 128,
    learning_rate: float = 0.001,
):
    """Train a model.

    Args:
        batch_size: Training batch size
        learning_rate: Initial learning rate for optimizer
    """
    pass
```

**Pros:**
- Supports multi-line descriptions
- Standard Python documentation style

**Cons:**
- Documentation separated from parameter definition
- More verbose

### Combining Both Syntaxes

If you provide both inline comments and docstring Args, params-proto will concatenate them on separate lines:

```python
@proto.cli
def train(
    batch_size: int = 128,  # Training batch size
):
    """Train a model.

    Args:
        batch_size: Controls memory usage and gradient noise
    """
    pass
```

This generates help text with multiple lines:

```
--batch-size INT     Training batch size
                     Controls memory usage and gradient noise (default: 128)
```

Use this pattern to provide:
- **Inline comment**: Brief, one-line summary
- **Docstring Args**: Detailed explanation with context

**Deduplication**: If both sources contain identical text, it will only appear once.

### 3. Auto-generated Descriptions (Fallback)

If no documentation is provided, params-proto auto-generates descriptions from parameter names:

```python
@proto.cli
def train(
    batch_size: int = 128,
    learning_rate: float = 0.001,
):
    """Train a model."""
    pass
```

Generates:
```
--batch-size INT     Batch size (default: 128)
--learning-rate FLOAT
                     Learning rate (default: 0.001)
```

## Supported Type Hints

### Basic Types

```python
@proto.cli
def example(
    count: int = 10,  # Integer
    ratio: float = 0.5,  # Float
    name: str = "model",  # String
    enabled: bool = True,  # Boolean flag
):
    pass
```

### Optional Types

```python
from typing import Optional


@proto.cli
def train(
    checkpoint: Optional[str] = None,  # Path to checkpoint
    resume_step: Optional[int] = None,  # Step to resume from
):
    pass
```

### List Types

```python
from typing import List


@proto.cli
def train(
    gpu_ids: List[int] = [0, 1],  # GPU device IDs
    data_dirs: List[str] = ["./data"],  # Data directories
):
    pass
```

### Literal Types (Choices)

```python
from typing import Literal


@proto.cli
def train(
    activation: Literal["relu", "gelu", "tanh"] = "relu",  # Activation function
    optimizer: Literal["adam", "sgd", "adamw"] = "adam",  # Optimizer type
):
    pass
```

### Tuple Types

```python
from typing import Tuple


@proto.cli
def train(
    image_size: Tuple[int, int] = (224, 224),  # Image height and width
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),  # RGB mean
):
    pass
```

### Enum Types

```python
from enum import Enum


class Optimizer(Enum):
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"


@proto.cli
def train(
    optimizer: Optimizer = Optimizer.ADAM,  # Optimizer algorithm
):
    pass
```

### Path Types

```python
from pathlib import Path


@proto.cli
def train(
    output_dir: Path = Path("./outputs"),  # Output directory
    data_root: Path = Path("/data"),  # Data root path
):
    pass
```

### Union Types

```python
from typing import Union


@proto.cli
def train(
    learning_rate: Union[float, str] = 0.001,  # Learning rate or 'auto'
    batch_size: Union[int, str] = 32,  # Batch size or 'auto'
):
    pass
```

## Environment Variables

Read configuration from environment variables with automatic type conversion:

```python
from params_proto import proto, EnvVar


@proto.cli
def train(
    # Three syntaxes for environment variables:
    batch_size: int = EnvVar @ "BATCH_SIZE",  # Read from env var
    learning_rate: float = EnvVar @ "LR" | 0.001,  # With default fallback
    db_url: str = EnvVar("DATABASE_URL", default="localhost"),  # Function syntax
):
    """Train with environment configuration."""
    pass
```

The pipe operator (`|`) provides clean syntax for fallback values:

```python
learning_rate: float = EnvVar @ "LR" | 0.001
```

Environment variables support template expansion:

```python
data_dir: str = EnvVar @ "$DATA_DIR/models"  # Expands $DATA_DIR
log_path: str = EnvVar @ "${LOG_DIR}/app.log"  # Expands ${LOG_DIR}
project_path: str = EnvVar @ "$BASE/$PROJECT"  # Multiple variables
```

## Best Practices

### 1. Use Type Hints

Always provide type hints for automatic CLI generation and validation:

```python
# Good
def train(batch_size: int = 128):
    pass

# Bad - no type hint
def train(batch_size=128):
    pass
```

### 2. Prefer Inline Comments

For single-line documentation, use inline comments:

```python
# Good
batch_size: int = 128,  # Training batch size

# Acceptable but verbose
batch_size: int = 128,
"""
Args:
    batch_size: Training batch size
"""
```

### 3. Use Descriptive Names

Parameter names should be self-explanatory:

```python
# Good
learning_rate: float = 0.001,  # Initial learning rate

# Bad - unclear abbreviation
lr: float = 0.001,  # Learning rate
```

### 4. Group Related Parameters

Use `@proto.prefix` for hierarchical organization:

```python
@proto.prefix
class Model:
    """Model configuration."""
    name: str = "resnet50"  # Model architecture
    pretrained: bool = True  # Use pretrained weights


@proto.prefix
class Train:
    """Training hyperparameters."""
    lr: float = 0.001  # Learning rate
    batch_size: int = 32  # Batch size


@proto.cli
def main(seed: int = 42):  # Random seed
    """Train a model."""
    print(f"Training {Model.name} with lr={Train.lr}")
```

Command line usage:

```bash
python train.py --Model.name vit --Train.lr 0.0001 --seed 123
```

## Testing Your CLI

Override the script name for predictable help output in tests:

```python
@proto.cli(prog="train.py")
def train(batch_size: int = 128):
    """Train a model."""
    pass


# In tests
def test_help():
    expected = """
    usage: train.py [-h] [--batch-size INT]
    ...
    """
    assert train.__help_str__ == expected
```

This ensures consistent help text regardless of how tests are run (pytest, IDE runner, etc.).
