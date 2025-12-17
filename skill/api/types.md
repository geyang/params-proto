---
title: Type Annotations
description: Supported type hints for CLI parsing and help generation
---

# Type Annotations

params-proto uses type hints for CLI parsing and help generation.

## Basic Types

```python
@proto.cli
def example(
    count: int = 10,      # INT in help
    rate: float = 0.5,    # FLOAT in help
    name: str = "default",  # STR in help
    enabled: bool = True,  # BOOL in help
): ...
```

## Boolean Handling

```python
@proto.cli
def train(
    verbose: bool = False,  # --verbose BOOL (to enable)
    cuda: bool = True,      # --no-cuda BOOL (to disable)
): ...
```

- `default=False` → help shows `--flag`
- `default=True` → help shows `--no-flag`

CLI usage:
```bash
python train.py --verbose     # verbose=True
python train.py --no-cuda     # cuda=False
```

## Optional Types

```python
from typing import Optional

@proto.cli
def process(
    config: str | None = None,  # Python 3.10+
    path: Optional[str] = None,  # typing.Optional
): ...
```

## Union Types

### Simple Unions

```python
@proto.cli
def train(
    lr: int | float = 0.001,  # Accepts either
    seed: int | None = None,  # Optional int
): ...
```

### Dataclass Unions (Subcommand Pattern)

```python
from dataclasses import dataclass

@dataclass
class Adam:
    lr: float = 0.001
    beta1: float = 0.9

@dataclass
class SGD:
    lr: float = 0.01
    momentum: float = 0.9

@proto.cli
def train(optimizer: Adam | SGD):
    """First positional arg selects type."""
    print(f"Using {type(optimizer).__name__}")
```

```bash
python train.py adam --lr 0.001
python train.py sgd --momentum 0.95
```

Class names are converted to kebab-case:
- `PerspectiveCamera` → `perspective-camera`
- `HTTPServer` → `http-server`

## Literal Types

```python
from typing import Literal

@proto.cli
def train(
    optimizer: Literal["adam", "sgd", "rmsprop"] = "adam",
    precision: Literal[16, 32, 64] = 32,
): ...
```

Help shows: `--optimizer VALUE` with choices documented.

## Enum Types

```python
from enum import Enum

class Optimizer(Enum):
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"

@proto.cli
def train(
    optimizer: Optimizer = Optimizer.ADAM,
): ...
```

Help shows: `--optimizer {ADAM,SGD,RMSPROP}`

CLI usage:
```bash
python train.py --optimizer SGD
```

## Collection Types

### List

```python
from typing import List

@proto.cli
def process(
    files: List[str] = ["input.txt"],
    dims: List[int] = [128, 256],
): ...
```

```bash
python process.py --files a.txt b.txt c.txt
python process.py --dims 512 1024
```

### Tuple

```python
from typing import Tuple

@proto.cli
def train(
    image_size: Tuple[int, int] = (224, 224),
): ...
```

```bash
python train.py --image-size 256 256
```

## Path Types

```python
from pathlib import Path

@proto.cli
def process(
    input_dir: Path = Path("./data"),
    output: Path = Path("output.txt"),
): ...
```

Strings automatically converted to `Path` objects.

## Type Display in Help

| Python Type | CLI Display |
|-------------|-------------|
| `int` | `INT` |
| `float` | `FLOAT` |
| `str` | `STR` |
| `bool` | `BOOL` |
| `Enum` | `{MEMBER1,MEMBER2,...}` |
| Other | `VALUE` |

## Required vs Optional

```python
@proto.cli
def train(
    # Required: no default, type must be callable
    config: Params,

    # Optional: has default
    epochs: int = 100,
): ...
```

Required parameters:
- Show `(required)` in help
- Can be passed positionally
- Type hint is **called** to instantiate

## Type Conversion

Values from CLI are automatically converted:

```python
@proto.cli
def train(
    lr: float = 0.001,  # "0.01" → 0.01
    epochs: int = 100,  # "50" → 50
    debug: bool = False,  # flag → True
): ...
```

Boolean string conversion:
- True: `"true"`, `"1"`, `"yes"`, `"on"`
- False: `"false"`, `"0"`, `"no"`, `"off"`

## Best Practices

1. **Be specific** - Use `Literal` or `Enum` over `str` when possible
2. **Use Optional explicitly** - `str | None` not `str = None`
3. **Document constraints** - In comments: `# Learning rate (0, 1)`
4. **Prefer Enum for fixed sets** - Better IDE support and validation
