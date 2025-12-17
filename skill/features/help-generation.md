---
title: Help Text Generation
description: Automatic CLI help text from inline comments and docstrings
---

# Help Text Generation

params-proto automatically generates CLI help text from code.

## Sources of Help Text

### 1. Inline Comments

```python
@proto.cli
def train(
    lr: float = 0.001,  # Learning rate for optimizer
    batch_size: int = 32,  # Training batch size
):
    pass
```

### 2. Docstring Description

```python
@proto.cli
def train(lr: float = 0.001):
    """Train a neural network on CIFAR-10.

    This trains a ResNet model using the specified hyperparameters.
    """
```

### 3. Docstring Args Section

```python
@proto.cli
def train(lr: float = 0.001):
    """Train a model.

    Args:
        lr: Learning rate controlling gradient step size
    """
```

### 4. Combined (Multi-line Help)

```python
@proto.cli
def train(
    lr: float = 0.001,  # Learning rate
):
    """Train a model.

    Args:
        lr: Controls optimization step size
    """
```

Output:
```
--lr FLOAT    Learning rate
              Controls optimization step size (default: 0.001)
```

## Help Text Elements

### Usage Line

```
usage: script.py [-h] [--lr FLOAT] [--batch-size INT]
```

- Shows program name
- Lists all arguments with types
- Boolean flags show `--flag` or `--no-flag` based on default

### Description

From the function/class docstring (first paragraph).

### Options Section

```
options:
  -h, --help           show this help message and exit
  --lr FLOAT           Learning rate (default: 0.001)
  --batch-size INT     Training batch size (default: 32)
```

### Prefix Sections

For `@proto.prefix` classes:

```
Model options:
  Model configuration.

  --model.name STR     Architecture name (default: resnet50)
  --model.layers INT   Number of layers (default: 50)
```

## Type Display

| Type | Display |
|------|---------|
| `int` | `INT` |
| `float` | `FLOAT` |
| `str` | `STR` |
| `bool` | `BOOL` |
| `Enum` | `{MEMBER1,MEMBER2}` |
| Other | `VALUE` |

## Default Values

```
--lr FLOAT    Learning rate (default: 0.001)
--seed INT    Random seed (required)
```

- Optional params show `(default: value)`
- Required params show `(required)`

## Boolean Flag Display

```python
@proto.cli
def train(
    verbose: bool = False,  # Enable verbose logging
    cuda: bool = True,  # Use CUDA acceleration
): ...
```

Output:
```
--verbose BOOL       Enable verbose logging (default: False)
--no-cuda BOOL       Use CUDA acceleration (default: True)
```

- `default=False` → shows `--flag`
- `default=True` → shows `--no-flag`

## ANSI Colors (Terminal)

When running in a terminal, help text is colorized:

- **Type names** (INT, FLOAT, etc.) → Bold bright blue
- **(required)** → Bold red
- **(default: value)** → Cyan with bold value

## Customizing Program Name

```python
@proto.cli(prog="my-trainer")
def train(): ...
```

Shows `usage: my-trainer` instead of filename.

## Auto-generated Descriptions

If no comment or docstring arg:

```python
@proto.cli
def train(
    learning_rate: float = 0.001,  # No comment
): ...
```

Generates: `Learning rate (default: 0.001)`

- Underscores → spaces
- First letter capitalized

## Long Parameter Names

```python
@proto.cli
def train(
    this_is_a_very_long_parameter_name: int = 100,  # Description
): ...
```

Output wraps:
```
--this-is-a-very-long-parameter-name INT
                     Description (default: 100)
```

## Best Practices

1. **Write clear inline comments** - Primary source of help text
2. **Keep descriptions concise** - One line ideally
3. **Document constraints** - "Learning rate in (0, 1)"
4. **Use docstrings for overview** - Explain what the command does
5. **Test with --help** - Verify output is clear
