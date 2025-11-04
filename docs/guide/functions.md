# Function-Based Configurations

> **Note**: This page is under construction. Check back soon for complete documentation.

Function-based configurations allow you to define parameters as function arguments with type hints.

## Basic Example

```python
from params_proto import proto

@proto
def train(
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Batch size
):
    """Train a model."""
    print(f"Training with lr={lr}")
```

See [Decorators](decorators.md) for more details.
