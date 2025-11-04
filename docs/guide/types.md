# Type Annotations

> **Note**: This page is under construction. Check back soon for complete documentation.

params-proto supports rich type annotations including Union types, Enums, Literals, and tuples.

## Supported Types

- `int`, `float`, `str`, `bool`
- `Union` types: `int | float`
- `Literal` types: `Literal["a", "b", "c"]`
- `Enum` types
- `tuple` types: `tuple[int, int]`
- Optional types: `str | None`

## Example

```python
from typing import Literal
from enum import Enum, auto

class Optimizer(Enum):
    ADAM = auto()
    SGD = auto()

@proto
class Config:
    precision: Literal["fp16", "fp32"] = "fp32"
    optimizer: Optimizer = Optimizer.ADAM
```
