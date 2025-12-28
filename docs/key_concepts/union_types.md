# Union Types and Subcommands

Quick reference for working with Union types and the difference between `Union[T]` and `Optional[T]`.

## Quick Reference

```python
from dataclasses import dataclass
from params_proto import proto

# Pattern 1: Union of multiple classes (choose one)
@dataclass
class Adam:
    lr: float = 0.001

@dataclass
class SGD:
    lr: float = 0.001
    momentum: float = 0.9

@proto.cli
def train(optimizer: Adam | SGD, epochs: int = 10):
    pass

# CLI: python train.py --optimizer:Adam --optimizer.lr 0.01
# CLI: python train.py adam --epochs 50
```

```python
# Pattern 2: Single class parameter (configuration)
@dataclass
class Model:
    name: str = "resnet50"
    hidden_size: int = 256

@proto.cli
def main(model: Model):
    pass

# CLI: python main.py --model:Model --model.hidden-size 512
# CLI: python main.py model
```

```python
# Pattern 3: Optional simple parameters (workaround)
# Note: Optional[str] is not fully supported; use str with default instead
@proto.cli
def process(checkpoint: str = None, batch_size: int = 32):
    pass

# CLI: python process.py --checkpoint model.pt
# CLI: python process.py --checkpoint model.pt --batch-size 64
```

## Union[ClassA, ClassB]: Choosing Classes

Use Union types when you need to **select which class to instantiate**:

```python
@dataclass
class PerspectiveCamera:
    fov: float = 60.0
    aspect: float = 1.33

@dataclass
class OrthographicCamera:
    scale: float = 1.0

@proto.cli
def render(camera: PerspectiveCamera | OrthographicCamera, output: str = "out.png"):
    pass
```

**CLI Syntax:**

```bash
# Named selection (all these work)
python render.py --camera:PerspectiveCamera
python render.py --camera:perspective-camera
python render.py --camera:perspectivecamera

# Positional selection
python render.py perspective-camera

# With attribute overrides
python render.py perspective-camera --camera.fov 45 --camera.aspect 1.77
```

## Optional[T]: Simple Optional Parameters

`Optional[T]` is for parameters that **may or may not be provided**:

```python
@proto.cli
def train(
    checkpoint: str = None,      # Works (workaround)
    # checkpoint: Optional[str] = None,  # ⚠️ Doesn't fully work yet
    epochs: int = 10,
):
    """Train model."""
    pass
```

**Expected CLI usage:**

```bash
python train.py --checkpoint model.pt       # Provide value
python train.py                             # Omit for None default
python train.py --checkpoint model.pt --epochs 50
```

```{note}
**Current limitation:** `Optional[str]`, `Optional[int]`, etc. are not fully supported.
Use regular parameters with defaults as a workaround:

```python
# ✓ Works
@proto.cli
def train(checkpoint: str = None, epochs: int = 10):
    pass

# ⚠️ Doesn't work yet
@proto.cli
def train(checkpoint: Optional[str] = None, epochs: int = 10):
    pass
```
```

## Key Differences

| Type | Purpose | CLI Syntax | When to Use |
|------|---------|-----------|-------------|
| `Union[ClassA, ClassB]` | Choose which class instance | `--param:ClassName` or positional | Multiple configurations (optimizers, models, etc.) |
| `Optional[str]` | Value may or may not be provided | `--param value` | Optional simple parameters (**currently use workaround**) |
| `str` with default | Same as Optional | `--param value` | Simple optional parameters (**recommended workaround**) |

## Syntax Variations

**Class selection** (all work):

```bash
--camera:PerspectiveCamera      # Exact match
--camera:perspective-camera     # kebab-case
--camera:perspectivecamera      # lowercase
perspective-camera              # Positional
```

**Attribute overrides** (kebab-case conversion):

```python
@dataclass
class Config:
    batch_size: int = 32          # Python: snake_case
    learning_rate: float = 0.001

# CLI uses kebab-case
--config.batch-size 64
--config.learning-rate 0.01
```

## Examples

### Example 1: Optimizer Selection

```python
@dataclass
class Adam:
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999

@dataclass
class SGD:
    lr: float = 0.001
    momentum: float = 0.9

@proto.cli
def train(optimizer: Adam | SGD):
    if isinstance(optimizer, Adam):
        print(f"Adam: lr={optimizer.lr}, beta1={optimizer.beta1}")
    else:
        print(f"SGD: lr={optimizer.lr}, momentum={optimizer.momentum}")
```

**Usage:**

```bash
python train.py adam --optimizer.lr 0.01
python train.py sgd --optimizer.momentum 0.95
```

### Example 2: Configuration Class

```python
@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    user: str = "admin"

@proto.cli
def connect(db: DatabaseConfig):
    print(f"Connecting to {db.host}:{db.port}")
```

**Usage:**

```bash
python connect.py database-config --db.host prod.example.com --db.port 3306
```

### Example 3: Mixed Parameters

```python
@dataclass
class PerspectiveCamera:
    fov: float = 60.0

@dataclass
class OrthographicCamera:
    scale: float = 1.0

@proto.cli
def render(
    camera: PerspectiveCamera | OrthographicCamera,
    output: str = "render.png",
    verbose: bool = False,
):
    """Render scene with selected camera."""
    pass
```

**Usage:**

```bash
python render.py perspective-camera --output scene.png --verbose
python render.py --camera:OrthographicCamera --camera.scale 2.0 --verbose
```

## Related

- [CLI Guide](cli_guide.md) - Full CLI documentation
- [Configuration Basics](configuration_basics.md) - Defining configurations
- [Type System](types.md) - Type annotation reference
