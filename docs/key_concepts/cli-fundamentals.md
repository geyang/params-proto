# CLI Fundamentals

`params-proto` automatically generates command-line interfaces from your Python code.

## How It Works

When you use `@proto.cli`, params-proto:

1. **Inspects your function signature** - Reads parameters, types, and defaults
2. **Extracts documentation** - From inline comments and docstrings
3. **Converts names** - Transforms Python naming to CLI conventions (snake_case → kebab-case)
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

usage: train.py [-h] [--learning-rate FLOAT] [--batch-size INT]

Train a model.

options:
  -h, --help             show this help message and exit
  --learning-rate FLOAT  Learning rate (default: 0.001)
  --batch-size INT       Batch size (default: 32)
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

```
--lr FLOAT      Learning rate (default: 0.001)
--epochs INT    Number of epochs (default: 100)
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

```
--data-path STR  Data path (required)
--lr FLOAT       Learning rate (default: 0.001)
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

```
usage: main.py [-h] [--seed INT] [OPTIONS]

Train model.

options:
  -h, --help              show this help message and exit
  --seed INT              Random seed (default: 42)

Model options:
  Model architecture.

  --Model.name STR        Model name (default: resnet50)
  --Model.hidden-size INT Hidden size (default: 256)

Training options:
  Training hyperparameters.

  --Training.lr FLOAT        Learning rate (default: 0.001)
  --Training.batch-size INT  Batch size (default: 32)
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

```
usage: train_model [-h] [--lr FLOAT]
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

## Union and Class Parameters

See [Union Types](union-types.md) for:
- Union subcommands (choosing between multiple configurations)
- Single class parameters
- Attribute overrides
- Positional selection

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

```
--this-is-a-very-long-parameter-name INT
                     This is a very long parameter name (default: 1000)
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

## Related

- [Naming Conventions](naming-conventions.md) - Parameter and class name conversion rules
- [Help Generation](help-generation.md) - Documentation extraction and formatting
- [Union Types](union-types.md) - Union subcommands and optional parameters
- [Type System](type-system.md) - Supported types and conversion
- [ANSI Formatting](ansi-formatting.md) - Terminal colors and formatting
