# Union Types: Subcommands and Configuration Selection

Union types enable powerful multi-way dispatching in your CLI programs, allowing users to choose between different configurations or implementations at runtime. This is essential for building flexible, composable CLI tools.

## Why Union Types Matter

When building complex CLI applications, you often need to support **multiple alternatives**:
- Different optimizer algorithms (Adam vs SGD vs RMSprop)
- Different model architectures (ResNet vs Vision Transformer vs CNN)
- Different data loaders or preprocessing strategies
- Different deployment environments (local vs cloud vs GPU)

Without Union types, you'd need separate scripts or manual conditional logic. With Union types, params-proto automatically generates:
- ✅ Subcommand-like syntax that feels natural
- ✅ Isolated parameter spaces (each option has its own parameters)
- ✅ Type-safe dispatch with `isinstance()` checks
- ✅ Automatic help generation per option

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

# CLI: python train.py adam --lr 0.01 --epochs 50
# CLI: python train.py sgd --momentum 0.95
```

```{note}
**Version 3.2.0+**: Subcommand attributes are **unprefixed by default**.
Use `--lr` instead of `--optimizer.lr`. The prefixed syntax still works for backwards compatibility.
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

# CLI: python main.py model --hidden-size 512
# CLI: python main.py model --name vit
```

```python
# Pattern 3: Optional simple parameters
from typing import Optional

@proto.cli
def process(checkpoint: Optional[str] = None, batch_size: int = 32):
    pass

# CLI: python process.py --checkpoint model.pt
# CLI: python process.py --checkpoint model.pt --batch-size 64
```

## Union[ClassA, ClassB]: Choosing Classes

Use Union types when you need to **select which class to instantiate**. This creates subcommand-like behavior where each option has isolated parameters.

### Example: Camera Selection

First, let's see how users interact with this from the command line:

```bash
# Positional selection with unprefixed attrs (v3.2.0+)
python render.py perspective-camera --fov 45 --aspect 1.77

# Alternative class
python render.py orthographic-camera --scale 2.0

# With shared parameters
python render.py perspective-camera --output scene.png

# Prefixed syntax still works (backwards compatible)
python render.py --camera:perspective-camera --camera.fov 45
```

Now the implementation:

```python
from dataclasses import dataclass
from params_proto import proto

@dataclass
class PerspectiveCamera:
    fov: float = 60.0          # Field of view in degrees
    aspect: float = 1.33       # Aspect ratio

@dataclass
class OrthographicCamera:
    scale: float = 1.0         # Orthographic scale factor

@proto.cli
def render(
    camera: PerspectiveCamera | OrthographicCamera,  # Union type creates selector
    output: str = "out.png",                           # Shared parameter
):
    """Render scene with selected camera type."""
    if isinstance(camera, PerspectiveCamera):
        print(f"Rendering with perspective: fov={camera.fov}, aspect={camera.aspect}")
    else:
        print(f"Rendering with orthographic: scale={camera.scale}")

if __name__ == "__main__":
    render()
```

**How it works:**
- `camera` is a **required parameter** with Union type
- params-proto instantiates the selected class
- Attributes are unprefixed by default (`--fov`, `--aspect`)
- Use `isinstance()` to dispatch on the selected type

## Optional[T]: Simple Optional Parameters

`Optional[T]` is for parameters that **may or may not be provided**:

```python
from typing import Optional

@proto.cli
def train(
    checkpoint: Optional[str] = None,  # Optional with None default
    epochs: int = 10,
):
    """Train model."""
    pass
```

**CLI usage:**

```bash
python train.py --checkpoint model.pt       # Provide value
python train.py                             # Omit for None default
python train.py --checkpoint model.pt --epochs 50
```

Both `Optional[str]` and `str = None` work equivalently:

```python
# Both work
@proto.cli
def train(checkpoint: Optional[str] = None): ...

@proto.cli
def train(checkpoint: str = None): ...
```

## Key Differences

| Type | Purpose | CLI Syntax | When to Use |
|------|---------|-----------|-------------|
| `Union[ClassA, ClassB]` | Choose which class instance | `--param:ClassName` or positional | Multiple configurations (optimizers, models, etc.) |
| `Optional[str]` | Value may or may not be provided | `--param value` | Optional simple parameters |
| `str = None` | Same as Optional | `--param value` | Alternative syntax for optional parameters |

## Syntax Variations

**Class selection** (all work):

```bash
--camera:PerspectiveCamera      # Exact match
--camera:perspective-camera     # kebab-case
--camera:perspectivecamera      # lowercase
perspective-camera              # Positional (recommended)
```

**Attribute overrides** (v3.2.0+: unprefixed by default):

```python
@dataclass
class Config:
    batch_size: int = 32          # Python: snake_case
    learning_rate: float = 0.001

# CLI uses kebab-case (unprefixed by default)
--batch-size 64
--learning-rate 0.01

# Prefixed syntax still works
--config.batch-size 64
--config.learning-rate 0.01
```

### Using `@proto.prefix` for Required Prefixes

If you want to **require** prefixed syntax for a class (e.g., for disambiguation), decorate it with `@proto.prefix`:

```python
@proto.prefix  # Now requires --config.batch-size syntax
@dataclass
class Config:
    batch_size: int = 32

@dataclass
class Model:
    batch_size: int = 16  # Same attr name, no conflict

@proto.cli
def train(config: Config, model: Model):
    pass

# CLI: Config requires prefix, Model doesn't
# python train.py config model --config.batch-size 64 --batch-size 32
```

## Examples

### Example 1: Optimizer Selection

Command-line usage:

```bash
# Choose optimizer and override defaults (unprefixed)
python train.py adam --lr 0.01
python train.py sgd --momentum 0.95

# Prefixed syntax also works
python train.py --optimizer:Adam --optimizer.beta1 0.95
```

Implementation:

```python
@dataclass
class Adam:
    lr: float = 0.001        # Learning rate
    beta1: float = 0.9       # Exponential decay rate for 1st moment
    beta2: float = 0.999     # Exponential decay rate for 2nd moment

@dataclass
class SGD:
    lr: float = 0.001        # Learning rate
    momentum: float = 0.9    # Momentum factor

@proto.cli
def train(optimizer: Adam | SGD):  # Union type selector
    """Train with chosen optimizer."""
    if isinstance(optimizer, Adam):
        print(f"Adam: lr={optimizer.lr}, beta1={optimizer.beta1}, beta2={optimizer.beta2}")
    elif isinstance(optimizer, SGD):
        print(f"SGD: lr={optimizer.lr}, momentum={optimizer.momentum}")
```

### Example 2: Configuration Class

Command-line usage:

```bash
# Positional class selection with unprefixed attrs
python connect.py database-config --host prod.example.com --port 3306

# Or named selection with prefixed attrs
python connect.py --db:DatabaseConfig --db.user root
```

Implementation:

```python
@dataclass
class DatabaseConfig:
    host: str = "localhost"  # Database host
    port: int = 5432         # Database port
    user: str = "admin"      # Database user

@proto.cli
def connect(db: DatabaseConfig):  # Single class (still uses Union mechanism)
    """Connect to database with configuration."""
    print(f"Connecting to {db.host}:{db.port} as {db.user}")
```

### Example 3: Mixed Union and Regular Parameters

Command-line usage:

```bash
# Union option + shared parameters (unprefixed)
python render.py perspective-camera --fov 45 --output scene.png --verbose

# Different union option with overrides
python render.py orthographic-camera --scale 2.0 --verbose

# Help shows all options
python render.py --help
```

Implementation:

```python
@dataclass
class PerspectiveCamera:
    fov: float = 60.0        # Field of view in degrees

@dataclass
class OrthographicCamera:
    scale: float = 1.0       # Scale factor

@proto.cli
def render(
    camera: PerspectiveCamera | OrthographicCamera,  # Union selector
    output: str = "render.png",                        # Shared parameter
    verbose: bool = False,                             # Shared flag
):
    """Render scene with selected camera."""
    if isinstance(camera, PerspectiveCamera):
        print(f"Perspective render: fov={camera.fov}, output={output}")
    else:
        print(f"Orthographic render: scale={camera.scale}, output={output}")

    if verbose:
        print("Verbose output enabled")
```

**Key point:** Union parameters are **required**, while other parameters can be optional with defaults.

### Example 4: @proto.prefix Classes Require Prefixed Syntax

When a Union class is decorated with `@proto.prefix`, its CLI attributes require prefixed syntax:

```python
@proto.prefix  # Requires prefixed syntax
@dataclass
class Train:
    lr: float = 0.001
    epochs: int = 100

@dataclass  # Regular dataclass - unprefixed works
class Evaluate:
    checkpoint: str = "model.pt"

@proto.cli
def main(mode: Train | Evaluate):
    if isinstance(mode, Train):
        print(f"Training: lr={mode.lr}, epochs={mode.epochs}")
    else:
        print(f"Evaluating: {mode.checkpoint}")

if __name__ == "__main__":
    main()
```

Command-line usage:

```bash
# Train is @proto.prefix - requires prefixed syntax
python main.py train --mode.lr 0.01 --mode.epochs 50

# Evaluate is regular dataclass - unprefixed works
python main.py evaluate --checkpoint best.pt
```

This is useful when:
- You have multiple Union params with overlapping attribute names
- You want explicit namespacing for clarity
- You're using the class as a singleton config elsewhere

## Related

- [Core Concepts](core-concepts) - Three main decorators
- [CLI Fundamentals](cli-fundamentals) - Basic CLI features
- [CLI Patterns](cli-patterns) - Advanced CLI patterns
- [Configuration Patterns](configuration-patterns) - Function vs class configurations
- [Type System](type-system) - Type annotation reference
