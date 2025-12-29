# Type Annotations

params-proto v3 supports rich type annotations for parameters, providing type safety and automatic CLI help generation.

## Known Type System Issues

⚠️ **The following types have CLI parsing issues in v3.0.0-rc21:**
- **`Tuple[T, ...]`** - Fixed-size tuples not fully supported
- **`Path`** - Strings not converted to Path objects
- **`dict`** - Collection types not implemented for CLI parsing
- **`Literal[...]` and `Enum`** - Help text works, but no runtime validation/conversion

These are documented in the [Type Support Matrix](#type-support-matrix) below. **Workarounds:** use `str` with documented defaults, or track the issue status.

## Required Parameters and Callable Types

**Key Design Principle:** For required parameters (those without default values), params-proto **always calls the type hint as a constructor**.

This is the foundation for Union types and potential subcommand support.

### How It Works

When a parameter has no default value, its type hint must be **callable**:

```python
from dataclasses import dataclass
from params_proto import proto

@dataclass
class Params:
    lr: float = 0.001
    batch_size: int = 32

@proto.cli
def train(
    config: Params,  # Required parameter - Params will be called as Params()
    epochs: int = 100,  # Optional parameter with default
):
    """Train with configuration."""
    print(f"Using lr={config.lr}, batch_size={config.batch_size}")
```

**CLI Usage:**
```bash
# params-proto calls Params() to instantiate it
python train.py params --lr 0.01 --batch-size 64
```

### Union Types as Subcommands

This pattern enables **Union types to act like subcommands**:

```python
from dataclasses import dataclass

@dataclass
class Perspect:
    """Perspective camera configuration."""
    fov: float = 60.0  # Field of view in degrees
    near: float = 0.1  # Near clipping plane
    far: float = 100.0  # Far clipping plane

@dataclass
class Orthographic:
    """Orthographic camera configuration."""
    zoom: float = 1.0  # Zoom level
    near: float = 0.1  # Near clipping plane
    far: float = 100.0  # Far clipping plane

@proto.cli
def render(
    camera: Perspect | Orthographic,  # Required - user must choose which type
    output: str = "output.png",
):
    """Render scene with camera configuration."""
    print(f"Using camera: {camera}")
```

**CLI Usage:**
```bash
# Choose perspective camera (calls Perspect())
python render.py perspect --fov 45.0 --near 0.1

# Choose orthographic camera (calls Orthographic())
python render.py orthographic --zoom 2.0 --near 0.5

# Get help for specific camera type
python render.py perspect --help
python render.py orthographic --help
```

The first positional argument selects **which type to instantiate**, and subsequent arguments configure that instance.

### Why This Matters

This design enables:

1. **Type-safe subcommand patterns** without special subcommand syntax
2. **Polymorphic configurations** - choose between different config types at runtime
3. **Composable types** - any callable (dataclass, class, function) works as a type hint
4. **Automatic CLI generation** - params-proto generates appropriate help text for each union member

### Required vs Optional

The **callable type instantiation only applies to required parameters**:

```python
@proto.cli
def train(
    # Required: Type hint MUST be callable, will be instantiated
    config: Params,

    # Optional: Uses the default value, type hint is for validation
    epochs: int = 100,
    lr: float = 0.001,
):
    pass
```

For optional parameters (with defaults), the type hint is used for **type conversion and validation**, not instantiation.

### Class Name to CLI Command Conversion

**Important:** When using Union types, class names are converted to lowercase CLI commands:

```python
from dataclasses import dataclass

@dataclass
class Perspect:      # Python: PascalCase
    fov: float = 60.0

@dataclass
class Orthographic:
    zoom: float = 1.0

@proto.cli
def render(camera: Perspect | Orthographic):
    """Render with camera."""
    pass
```

**CLI usage:**
```bash
# Class names become kebab-case commands
python render.py perspect --fov 45.0       # Not Perspect or PERSPECT
python render.py orthographic --zoom 2.0   # Not Orthographic
```

**Conversion rule:** PascalCase → kebab-case (e.g., `HTTPServer` → `http-server`, `MLModel` → `ml-model`)

#### Best Practices for Union Type Names

**Use simple single-word names:**

```python
# ✓ Recommended
class Train:      # → train
class Evaluate:   # → evaluate
class Export:     # → export
```

**Acronyms and multi-word names now convert properly:**

```python
# ✓ Good: acronyms now convert to kebab-case
class HTTPServer:    # → http-server
class MLModel:       # → ml-model
class DeepQNetwork:  # → deep-q-network

# ✓ Also good: simple alternatives remain clear
class Server:     # → server
class Model:      # → model
class Network:    # → network
```

## Basic Types

### Primitive Types

```python
from params_proto import proto

@proto.cli
def example(
    count: int = 10,  # Integer parameter
    rate: float = 0.5,  # Floating point parameter
    name: str = "default",  # String parameter
    enabled: bool = True,  # Boolean flag
):
    """Example with basic types."""
    pass
```

**CLI usage:**
```bash
python example.py --count 20 --rate 0.75 --name custom --enabled
python example.py --no-enabled  # Set to False
```

### Type Conversion

Values are automatically converted to the annotated type:

```python
@proto.cli
def train(
    lr: float = 0.001,  # Converts "0.01" → 0.01
    epochs: int = 100,  # Converts "50" → 50
    debug: bool = False,  # Converts "true" → True
):
    pass
```

**Boolean conversion:**
- `True`: `"true"`, `"1"`, `"yes"`, `"on"` (case-insensitive)
- `False`: `"false"`, `"0"`, `"no"`, `"off"`, or flag `--no-{param}`

## Optional Types

### Using `| None` (Python 3.10+)

```python
@proto.cli
def process(
    config_file: str | None = None,  # Optional string
    max_items: int | None = None,  # Optional integer
):
    """Process with optional parameters."""
    if config_file:
        print(f"Using config: {config_file}")
    else:
        print("Using defaults")
```

### Using `Optional` (Python 3.9+)

```python
from typing import Optional

@proto.cli
def process(
    config_file: Optional[str] = None,
    max_items: Optional[int] = None,
):
    """Same as above, alternative syntax."""
    pass
```

## Union Types

### Multiple Accepted Types

```python
@proto.cli
def train(
    lr: int | float = 0.001,  # Accepts either int or float
    seed: int | None = None,  # Accepts int or None
):
    """Training with union types."""
    pass
```

**CLI usage:**
```bash
python train.py --lr 0.01  # float
python train.py --lr 1  # int (also valid)
```

## Literal Types

### Restricted Value Sets

```python
from typing import Literal

@proto.cli
def train(
    optimizer: Literal["adam", "sgd", "rmsprop"] = "adam",  # Only these values
    device: Literal["cuda", "cpu", "mps"] = "cuda",  # Hardware selection
):
    """Training with literal types."""
    print(f"Using {optimizer} on {device}")
```

**CLI usage:**
```bash
python train.py --optimizer sgd --device cpu
# python train.py --optimizer invalid  # Would be an error
```

**Help text shows options:**
```{ansi-block}
:string_escape:

--optimizer {adam,sgd,rmsprop}
                     Optimizer \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36madam\x1b[0m\x1b[36m)\x1b[0m
```

## Enum Types

### Using Python Enums

```python
from enum import Enum, auto
from params_proto import proto

class Optimizer(Enum):
    """Optimizer choices."""
    ADAM = auto()
    SGD = auto()
    RMSPROP = auto()

class Device(Enum):
    """Hardware device."""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"

@proto.cli
def train(
    optimizer: Optimizer = Optimizer.ADAM,  # Enum parameter
    device: Device = Device.CUDA,  # Enum with custom values
):
    """Training with enum types."""
    print(f"Optimizer: {optimizer.name}")  # ADAM
    print(f"Device: {device.value}")  # cuda
```

**CLI usage:**
```bash
python train.py --optimizer SGD --device CPU
```

**Help text:**
```{ansi-block}
:string_escape:

--optimizer {ADAM,SGD,RMSPROP}
                     Optimizer \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36mADAM\x1b[0m\x1b[36m)\x1b[0m
```

## Collection Types

### List Types

List types allow collecting multiple values from the CLI. Each element is converted to the specified type.

```python
from typing import List

@proto.cli
def process(
    files: List[str] = ["input.txt"],  # List of strings
    dimensions: List[int] = [128, 256],  # List of integers
    ratios: List[float] = [0.5, 0.3],  # List of floats
):
    """Process multiple files."""
    print(f"Files: {files}")  # e.g., ['a.txt', 'b.txt', 'c.txt']
    print(f"Dimensions: {dimensions}")  # e.g., [512, 1024]
    print(f"Ratios: {ratios}")  # e.g., [0.8, 0.2]
```

**CLI usage:**
```bash
# Multiple string values
python process.py --files a.txt b.txt c.txt

# Multiple integer values (type-converted)
python process.py --dimensions 512 1024

# Multiple float values
python process.py --ratios 0.8 0.2

# Combine with other arguments
python process.py --files x.txt y.txt --dimensions 256 256 --ratios 0.5
```

**Help text shows list notation:**
```{ansi-block}
:string_escape:

--files [STR]            List of input files \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m['input.txt']\x1b[0m\x1b[36m)\x1b[0m
--dimensions [INT]       Dimensions to process \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m[128, 256]\x1b[0m\x1b[36m)\x1b[0m
--ratios [FLOAT]         Aspect ratios \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m[0.5, 0.3]\x1b[0m\x1b[36m)\x1b[0m
```

**How it works:**
- Arguments after `--flag` are collected until the next flag or end of arguments
- Each value is converted to the element type (e.g., `"256"` → `256` for `List[int]`)
- The result is always a list, even with a single value

### Tuple Types

```python
from typing import Tuple

@proto.cli
def train(
    image_size: Tuple[int, int] = (224, 224),  # Fixed-size tuple
    crop_size: tuple[int, int] = (112, 112),  # Python 3.9+ syntax
):
    """Training with tuple types."""
    print(f"Image size: {image_size[0]}x{image_size[1]}")
```

**CLI usage:**
```bash
python train.py --image-size 256 256
```

## Path Types

### pathlib.Path

```python
from pathlib import Path

@proto.cli
def process(
    input_dir: Path = Path("./data"),  # Path parameter
    output_file: Path = Path("output.txt"),  # File path
):
    """Process files with Path types."""
    print(f"Reading from: {input_dir}")
    print(f"Writing to: {output_file}")
```

**CLI usage:**
```bash
python process.py --input-dir /path/to/data --output-file results.txt
```

**Automatic conversion:**
- Strings are converted to `Path` objects
- Path validation happens at runtime (if you check)

## Complex Types

### Nested Unions

```python
@proto.cli
def train(
    lr: int | float | None = 0.001,  # Multiple options
    layers: List[int] | None = None,  # Optional list
):
    """Complex union types."""
    pass
```

### Literal with Union

```python
@proto.cli
def process(
    format: Literal["json", "yaml"] | None = None,  # Optional literal
    precision: Literal[16, 32, 64] = 32,  # Numeric literal
):
    """Literal with union types."""
    pass
```

## Type Support Matrix

| Type | CLI Support | Help Display | Notes |
|------|-------------|--------------|-------|
| `int` | ✅ Full | `INT` | Fully working |
| `float` | ✅ Full | `FLOAT` | Fully working |
| `str` | ✅ Full | `STR` | Fully working |
| `bool` | ✅ Full | (flag) | Supports `--flag` and `--no-flag` |
| `int \| float` | ✅ Full | `VALUE` | Ambiguous unions work |
| `str \| None` (Optional) | ✅ Full | `STR` | Correctly unwraps to inner type |
| `Literal[...]` | ⚠️ Partial | `{a,b,c}` | Help shows values, but no validation |
| `Enum` | ⚠️ Partial | `{A,B,C}` | Help shows members, no enum conversion |
| `List[T]` | ✅ Full | `[INT]` or `[STR]` | Fully working with element type conversion |
| `Tuple[T, ...]` | ❌ Broken | `VALUE` | **Not implemented for CLI parsing** |
| `Path` | ❌ Broken | `STR` | **Strings not converted to Path objects** |
| `dict` | ❌ Broken | `VALUE` | **Not implemented** |
| `Union[Class1, Class2]` | ✅ Full | Subcommand | Works as pseudo-subcommands |
| Custom classes | ❌ Broken | - | Must be dataclasses with Union context |

## Type Validation

### Runtime Checks

```python
@proto.cli
def train(lr: float = 0.001):
    """Training function."""
    # Type already converted by params-proto
    assert isinstance(lr, float), "lr must be float"

    # Add value validation
    if lr <= 0 or lr >= 1:
        raise ValueError("lr must be in (0, 1)")
```

### Using Dataclass Validation

```python
from dataclasses import dataclass
from params_proto import proto

@proto
@dataclass
class Params:    lr: float = 0.001
    batch_size: int = 32

    def __post_init__(self):
        """Validate after initialization."""
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
```

## Type Hints Best Practices

### 1. Be Specific

```python
# ✓ Good: specific literal
@proto.cli
def train(optimizer: Literal["adam", "sgd"] = "adam"):
    pass

# ✗ Avoid: too general
@proto.cli
def train(optimizer: str = "adam"):
    pass
```

### 2. Use Optional for Nullable Values

```python
# ✓ Good: explicit None
@proto.cli
def process(config: str | None = None):
    pass

# ✗ Avoid: implicit None
@proto.cli
def process(config: str = None):  # Type mismatch
    pass
```

### 3. Document Type Constraints

```python
# ✓ Good: documented constraints
@proto.cli
def train(
    lr: float = 0.001,  # Learning rate in range (0, 1)
    epochs: int = 100,  # Number of epochs (positive)
):
    """Training with documented types."""
    pass
```

### 4. Use Enums for Fixed Sets

```python
from enum import Enum

# ✓ Good: enum for options
class Model(Enum):
    RESNET = "resnet50"
    VIT = "vit-base"

@proto.cli
def train(model: Model = Model.RESNET):
    pass

# ✗ Avoid: magic strings
@proto.cli
def train(model: str = "resnet50"):
    # User could pass any string
    pass
```

## Advanced Type Patterns

### Generic Types

```python
from typing import TypeVar, Generic, List

T = TypeVar('T')

@proto
class Config(Generic[T]):
    """Generic configuration."""
    items: List[T]
    default: T

# Usage (type parameter determined at runtime)
int_config = Config[int](items=[1, 2, 3], default=0)
```

### NewType for Documentation

```python
from typing import NewType

# Create semantic types
LearningRate = NewType('LearningRate', float)
BatchSize = NewType('BatchSize', int)

@proto.cli
def train(
    lr: LearningRate = 0.001,  # Type hint is self-documenting
    batch_size: BatchSize = 32,
):
    """Training with semantic types."""
    pass
```

### Type Aliases

```python
from typing import Union, List

# Define complex types once
Numeric = Union[int, float]
PathLike = Union[str, Path]
StringList = List[str]

@proto.cli
def process(
    threshold: Numeric = 0.5,
    input_path: PathLike = "data",
    files: StringList = ["a.txt"],
):
    """Using type aliases."""
    pass
```

## Environment Variable Types

Type conversion also works with environment variables:

```python
from params_proto import proto, EnvVar

@proto.cli
def train(
    # Environment variables are type-converted
    lr: float = EnvVar @ "LEARNING_RATE" | 0.001,
    batch_size: int = EnvVar @ "BATCH_SIZE" | 32,
    use_cuda: bool = EnvVar @ "USE_CUDA" | True,
):
    """Types work with EnvVar."""
    pass
```

**Usage:**
```bash
LEARNING_RATE=0.01 BATCH_SIZE=64 python train.py
# lr will be float(0.01), batch_size will be int(64)
```

## Troubleshooting

### Type Mismatch Errors

```python
# Problem: None as default for non-optional type
@proto.cli
def bad(count: int = None):  # Type error!
    pass

# Solution: Use Optional
@proto.cli
def good(count: int | None = None):
    pass
```

### Union Type Ambiguity

```python
# Problem: Ambiguous union
@proto.cli
def ambiguous(value: int | str = 1):
    # CLI string "42" - is it int or str?
    pass

# Solution: Use Literal or Enum for clarity
@proto.cli
def clear(value: Literal[1, 2, 3] = 1):
    pass
```

## Related

- [Core Concepts](core-concepts) - Decorators and basic usage
- [CLI Fundamentals](cli-fundamentals) - Type display in CLI
- [CLI Patterns](cli-patterns) - Advanced patterns with type parameters
- [Union Types](union-types) - Union types as subcommands
- [Configuration Patterns](configuration-patterns) - Type annotations in configurations
