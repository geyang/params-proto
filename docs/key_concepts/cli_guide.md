# Building You CLI

`params-proto` automatically generates command-line interfaces from your Python code. There are two main ways to
do so: via a python class namespace, or via a function. Function interface is simple and useful for scripts,
but it lack the ability to directly reference and expose configuration objects. For more detailed discussion, refer to 
[Best Practice for Function Parameters](function_best_practices.md).

## How It Works

When you use `@proto.cli`, params-proto:

1. **Inspects your function signature** - Reads parameters, types, and defaults
2. **Extracts documentation** - From inline comments and docstrings
3. **Converts names** - Transforms Python naming to CLI conventions
4. **Generates argparse** - Creates CLI parser automatically
5. **Type converts** - Parses and validates CLI arguments

```python
from params_proto import proto


@proto.cli
def train(
  learning_rate: float = 0.001,  # Learning rate
  batch_size: int = 32,  # Batch size
):
  """Train a model."""
  pass
```

↓ **Automatically becomes** ↓

```bash
$ python train.py --help
```

```{ansi-block}
:string_escape:

usage: train.py [-h] [--learning-rate \x1b[1m\x1b[94mFLOAT\x1b[0m] [--batch-size \x1b[1m\x1b[94mINT\x1b[0m]

Train a model.

options:
  -h, --help            show this help message and exit
  --learning-rate \x1b[1m\x1b[94mFLOAT\x1b[0m Learning rate \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m0.001\x1b[0m\x1b[36m)\x1b[0m
  --batch-size \x1b[1m\x1b[94mINT\x1b[0m      Batch size \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m32\x1b[0m\x1b[36m)\x1b[0m
```

##naming Conventions

### Parameter Names: snake_case → kebab-case

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

### Class Names: PascalCase → lowercase

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
python tool.py train --lr 0.01     # Class: Train → command: train
python tool.py evaluate --model pt  # Class: Evaluate → command: evaluate
```

**Conversion rule:** Simple `.lower()` - no hyphen insertion

### Prefix Names: PascalCase Preserved

When using `@proto.prefix`, class names stay as-is:

```python
@proto.prefix
class Model:  # PascalCase preserved
  name: str = "resnet50"
  hidden_size: int = 256


@proto.prefix
class Training:  # PascalCase preserved
  lr: float = 0.001
```

**CLI arguments:**

```bash
--Model.name resnet50        # Prefix keeps PascalCase
--Model.hidden-size 512      # Parameter converts to kebab-case
--Training.lr 0.01           # Prefix keeps PascalCase
```

**Why?** Prefixes match Python code exactly for clarity:

```python
# In code
print(Model.name)  # PascalCase

# On CLI
--Model.name
resnet50  # PascalCase (matches code)
```

## Naming Best Practices

### 1. Use Simple Names for Union Types

```python
# ✓ Good: Simple single-word names
class Train:  # → train

  class Evaluate:  # → evaluate

  class Export:  # → export


# ⚠️ Problematic: Acronyms don't split
class HTTPServer:  # → httpserver (not http-server)

  class MLModel:  # → mlmodel (not ml-model)


# ✓ Better alternatives
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
class Model:  # --Model.param

  class Training:  # --Training.param


# ⚠️ Works but verbose
@proto.prefix
class DataLoader:  # --DataLoader.param (long)


# ✓ Better
@proto.prefix
class Data:  # --Data.param (shorter)
```

## Comparison with Other Tools

### params-proto: Simple Lowercase

```python
class HTTPServer:  # → httpserver

  class MLModel:  # → mlmodel

  class DeepQNetwork:  # → deepqnetwork
```

Simple `.lower()` conversion - no splitting.

### tyro: Aggressive Splitting

For comparison, [tyro](https://github.com/brentyi/tyro) splits every capital letter:

```python
# tyro's behavior (reference only)
class HTTPServer:  # → h-t-t-p-server (every capital split)

  class MLModel:  # → m-l-model

  class getAPIKey:  # → get-a-p-i-key
```

**params-proto uses simple conversion** to avoid surprises. Choose simple names that work well with lowercase.

## Help Text Generation

### Inline Comments

```python
@proto.cli
def train(
  lr: float = 0.001,  # This becomes the CLI help text
  batch_size: int = 32,  # Keep it short and descriptive
):
  pass
```

**Generated help:**

```{ansi-block}
:string_escape:

--lr \x1b[1m\x1b[94mFLOAT\x1b[0m           This becomes the CLI help text \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m0.001\x1b[0m\x1b[36m)\x1b[0m
--batch-size \x1b[1m\x1b[94mINT\x1b[0m     Keep it short and descriptive \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m32\x1b[0m\x1b[36m)\x1b[0m
```

### Docstring Args Section

```python
@proto.cli
def train(
  lr: float = 0.001,  # Learning rate
):
  """Train a model.

  Args:
      lr: Learning rate for the optimizer. Typical values are 0.001 for
          Adam and 0.01-0.1 for SGD. Reduce if training is unstable.
  """
  pass
```

**Generated help combines both:**

```{ansi-block}
:string_escape:

--lr \x1b[1m\x1b[94mFLOAT\x1b[0m           Learning rate
                     Learning rate for the optimizer. Typical values are 0.001
                     for Adam and 0.01-0.1 for SGD. Reduce if training is
                     unstable. \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m0.001\x1b[0m\x1b[36m)\x1b[0m
```

### Function Docstring

The function's main docstring becomes the CLI description:

```python
@proto.cli
def train(lr: float = 0.001):
  """Train a neural network on CIFAR-10.

  This function implements the full training loop including
  data loading, forward/backward passes, and checkpointing.
  """
  pass
```

**Generated help:**

```{ansi-block}
:string_escape:

usage: train.py [-h] [--lr \x1b[1m\x1b[94mFLOAT\x1b[0m]

Train a neural network on CIFAR-10.

options:
  ...
```

Only the first paragraph (before `Args:`) appears in help text.

### Auto-Generated Descriptions

If no documentation provided, params-proto generates basic descriptions from parameter names:

```python
@proto.cli
def train(
  learning_rate: float = 0.001,  # No comment
  batch_size: int = 32,  # No comment
):
  """Train a model."""
  pass
```

**Generated help:**

```{ansi-block}
:string_escape:

--learning-rate \x1b[1m\x1b[94mFLOAT\x1b[0m  Learning rate \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m0.001\x1b[0m\x1b[36m)\x1b[0m
--batch-size \x1b[1m\x1b[94mINT\x1b[0m       Batch size \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m32\x1b[0m\x1b[36m)\x1b[0m
```

## Type Display

Parameter types appear in help text:

| Python Type         | CLI Display | Example             |
|---------------------|-------------|---------------------|
| `int`               | `INT`       | `--count INT`       |
| `float`             | `FLOAT`     | `--lr FLOAT`        |
| `str`               | `STR`       | `--name STR`        |
| `bool`              | (flag)      | `--verbose`         |
| `int \| float`      | `VALUE`     | `--threshold VALUE` |
| `str \| None`       | `VALUE`     | `--config VALUE`    |
| `Literal["a", "b"]` | `VALUE`     | `--mode VALUE`      |
| `Enum`              | `{A,B,C}`   | `--opt {ADAM,SGD}`  |
| `List[int]`         | `VALUE`     | `--ids VALUE`       |
| `Path`              | `VALUE`     | `--dir VALUE`       |

## Boolean Flags

Boolean parameters become flags:

```python
@proto.cli
def train(
  verbose: bool = False,  # Flag (no argument)
  cuda: bool = True,  # Flag (no argument)
):
  pass
```

**CLI usage:**

```bash
# Set to True
python train.py --verbose

# Set to False
python train.py --no-verbose

# Boolean with True default
python train.py --cuda      # Still True (default)
python train.py --no-cuda   # Now False
```

## Required vs Optional

**Optional parameters** (with defaults):

```python
@proto.cli
def train(
  lr: float = 0.001,  # Optional
  epochs: int = 100,  # Optional
):
  pass
```

**Help text:**

```{ansi-block}
:string_escape:

--lr \x1b[1m\x1b[94mFLOAT\x1b[0m      Learning rate \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m0.001\x1b[0m\x1b[36m)\x1b[0m
--epochs \x1b[1m\x1b[94mINT\x1b[0m    Number of epochs \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m100\x1b[0m\x1b[36m)\x1b[0m
```

**Required parameters** (no defaults):

```python
@proto.cli
def train(
  data_path: str,  # Required!
  lr: float = 0.001,
):
  pass
```

**Help text:**

```{ansi-block}
:string_escape:

--data-path \x1b[1m\x1b[94mSTR\x1b[0m  Data path \x1b[1m\x1b[31m(required)\x1b[0m
--lr \x1b[1m\x1b[94mFLOAT\x1b[0m       Learning rate \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m0.001\x1b[0m\x1b[36m)\x1b[0m
```

## Grouped Options

`@proto.prefix` groups options in help text:

```python
@proto.prefix
class Model:
  """Model architecture."""
  name: str = "resnet50"
  hidden_size: int = 256


@proto.prefix
class Training:
  """Training hyperparameters."""
  lr: float = 0.001
  batch_size: int = 32


@proto.cli
def main(seed: int = 42):
  """Train model."""
  pass
```

**Generated help:**

```{ansi-block}
:string_escape:

usage: main.py [-h] [--seed \x1b[1m\x1b[94mINT\x1b[0m] [OPTIONS]

Train model.

options:
  -h, --help                   show this help message and exit
  --seed \x1b[1m\x1b[94mINT\x1b[0m                   Random seed \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m42\x1b[0m\x1b[36m)\x1b[0m

Model options:
  Model architecture.

  --Model.name \x1b[1m\x1b[94mSTR\x1b[0m             Model name \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36mresnet50\x1b[0m\x1b[36m)\x1b[0m
  --Model.hidden-size \x1b[1m\x1b[94mINT\x1b[0m      Hidden size \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m256\x1b[0m\x1b[36m)\x1b[0m

Training options:
  Training hyperparameters.

  --Training.lr \x1b[1m\x1b[94mFLOAT\x1b[0m          Learning rate \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m0.001\x1b[0m\x1b[36m)\x1b[0m
  --Training.batch-size \x1b[1m\x1b[94mINT\x1b[0m    Batch size \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m32\x1b[0m\x1b[36m)\x1b[0m
```

## Custom Program Name

Override the program name in help text:

```python
@proto.cli(prog="train_model")
def train(lr: float = 0.001):
  """Train a model."""
  pass
```

**Generated help:**

```{ansi-block}
:string_escape:

usage: train_model [-h] [--lr \x1b[1m\x1b[94mFLOAT\x1b[0m]
```

Without `prog`, uses script filename from `sys.argv[0]`.

## Testing Help Generation

Access help text programmatically:

```python
@proto.cli
def train(lr: float = 0.001):
  """Train a model."""
  pass


# Access generated help string
print(train.__help_str__)
```

Useful for testing and documentation generation.

## Edge Cases

### Long Parameter Names

```python
@proto.cli
def train(
  this_is_a_very_long_parameter_name: int = 1000,
):
  """Train with long names."""
  pass
```

**Generated help (wraps nicely):**

```{ansi-block}
:string_escape:

--this-is-a-very-long-parameter-name \x1b[1m\x1b[94mINT\x1b[0m
                     This is a very long parameter name \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m1000\x1b[0m\x1b[36m)\x1b[0m
```

### Numbers in Names

```python
@proto.cli
def train(
  model_2d: bool = False,  # → --model-2d
  resnet50_pretrained: bool = True,  # → --resnet50-pretrained
):
  pass
```

Numbers are preserved in CLI arguments.

## Summary

**Key conversions:**

- Parameters: `snake_case` → `--kebab-case`
- Union classes: `PascalCase` → `lowercase`
- Prefixes: `PascalCase` → `--PascalCase.kebab-case`
- Booleans: `bool` → `--flag` / `--no-flag`

**Best practices:**

- Use simple class names for Union types
- Use snake_case for all parameters
- Document with inline comments + docstrings
- Keep prefix names short and clear

## Related

- [Configuration Basics](configuration_basics.md) - Defining configs
- [Type System](types.md) - Type annotation reference
- [Advanced Patterns](advanced_patterns.md) - Prefixes and subcommands
- [Parameter Overrides](overrides.md) - CLI and programmatic overrides
