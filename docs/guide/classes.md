# Class-Based Configurations

> **Note**: This page is under construction. Check back soon for complete documentation.

Class-based configurations use Python classes with type-annotated attributes.

## Basic Example

```python
from params_proto import proto

@proto
class Config:
    """Training configuration."""
    lr: float = 0.001  # Learning rate
    batch_size: int = 32  # Batch size
```

See [Decorators](decorators.md) for more details.
