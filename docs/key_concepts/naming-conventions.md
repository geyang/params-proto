# Naming Conventions

params-proto automatically converts Python naming conventions to CLI conventions.

## Parameter Names: snake_case → kebab-case

Python parameters convert to CLI arguments:

```python
@proto.cli
def train(
  learning_rate: float = 0.001,  # Python: snake_case
  batch_size: int = 32,
  max_epochs: int = 100,
):
  pass
```

**CLI arguments:**

```bash
--learning-rate  # Converted to kebab-case
--batch-size
--max-epochs
```

**Conversion rule:** Replace `_` with `-`, then lowercase

## Class Names: PascalCase → kebab-case

When using Union types for subcommands, class names convert to CLI commands:

```python
from dataclasses import dataclass

@dataclass
class Train:  # PascalCase in Python
  lr: float = 0.001

@dataclass
class Evaluate:
  model: str

@proto.cli
def tool(command: Train | Evaluate):
  """Tool with subcommands."""
  pass
```

**CLI commands:**

```bash
python tool.py train --lr 0.01      # Class: Train → command: train
python tool.py evaluate --model pt  # Class: Evaluate → command: evaluate
```

**Conversion rule:** PascalCase → kebab-case (e.g., `HTTPServer` → `http-server`, `MLModel` → `ml-model`)

## Prefix Names: PascalCase → kebab-case

When using `@proto.prefix`, class names convert to kebab-case prefixes:

```python
@proto.prefix
class Model:  # PascalCase in Python
  name: str = "resnet50"
  hidden_size: int = 256

@proto.prefix
class Training:  # PascalCase in Python
  lr: float = 0.001
```

**CLI arguments:**

```bash
--model.name resnet50        # Prefix converts to kebab-case
--model.hidden-size 512      # Parameter converts to kebab-case
--training.lr 0.01           # Prefix converts to kebab-case
```

**Conversion rule:** Class name converts to kebab-case (e.g., `DataLoader` → `data-loader`), parameters convert to kebab-case.

```python
# In code
print(Model.name)  # PascalCase class

# On CLI
--model.name resnet50  # kebab-case prefix
```

## Naming Best Practices

### 1. Use Simple Names for Union Types

```python
# ✓ Good: Simple single-word names
class Train:  # → train
  pass

class Evaluate:  # → evaluate
  pass

class Export:  # → export
  pass

# ✓ Good: Acronyms now convert properly
class HTTPServer:  # → http-server
class MLModel:  # → ml-model
class DataLoader:  # → data-loader

# ✓ Also good: Simple single-word names
class Server:  # → server (simple and clear)
class Model:  # → model
```

### 2. Use snake_case for Parameters

```python
# ✓ Good: snake_case converts perfectly
@proto.cli
def train(
  learning_rate: float = 0.001,  # → --learning-rate
  batch_size: int = 32,  # → --batch-size
):
  pass

# ✗ Avoid: camelCase doesn't split
@proto.cli
def train(
  learningRate: float = 0.001,  # → --learningrate (no hyphen!)
):
  pass
```

### 3. Keep Prefix Names Simple

```python
# ✓ Good: Simple and clear
@proto.prefix
class Model:  # --model.param
  pass

@proto.prefix
class Training:  # --training.param
  pass

# ⚠️ Works but verbose
@proto.prefix
class DataLoader:  # --data-loader.param (long)
  pass

# ✓ Better
@proto.prefix
class Data:  # --data.param (shorter)
  pass
```

## Syntax Variations

**For Union type selection** (like `--camera:ClassName`), all naming conventions work:

| Python Class       | CLI Syntax Options                                              |
|--------------------|----------------------------------------------------------------|
| `PerspectiveCamera` | `perspective-camera`, `perspectivecamera`, `PerspectiveCamera` |
| `HTTPServer`       | `httpserver`, `http-server`, `HTTPServer`                       |
| `MLModel`          | `mlmodel`, `ml-model`, `MLModel`                                |

```{note}
For **prefix parameter access** (like `--http-server.port`), you must use the exact registered prefix name, which is the kebab-case version of the class name (e.g., `http-server` for `HTTPServer`).
```

**Attribute names** always convert to kebab-case:

```python
@dataclass
class Config:
    batch_size: int = 32      # → --config.batch-size
    learning_rate: float = 0.001  # → --config.learning-rate
```

## Summary

**Key conversions:**
- Parameters: `snake_case` → `--kebab-case`
- Union classes: `PascalCase` → `kebab-case`
- Prefixes: `PascalCase` → `--kebab-case.kebab-case`
- Booleans: `bool` → `--flag` / `--no-flag`

**Best practices:**
- Use simple class names for Union types
- Use snake_case for all parameters
- Keep prefix names short and clear
- Document with comments and docstrings

## Related

- [CLI Fundamentals](cli-fundamentals) - Core CLI features
- [CLI Patterns](cli-patterns) - Advanced CLI patterns
- [Help Generation](help-generation) - Documentation extraction
- [Union Types](union-types) - Union subcommands
