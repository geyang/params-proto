# CLI Patterns

Advanced patterns and techniques for building CLIs with params-proto.

## Grouped Options with Prefixes

Use `@proto.prefix` to organize related parameters into logical groups:

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

This groups options visually and makes help text more readable for complex CLIs with many parameters.

### Benefits

- **Organization** - Related parameters grouped together
- **Readability** - Help text clearly shows which parameters belong together
- **Discoverability** - Users can find related options easily
- **Scalability** - Works well with many parameters

See [Advanced Patterns](advanced-patterns) for more on prefixes and singleton behavior.

---

## Custom Program Name

Override the program name displayed in help text:

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

Without `prog`, uses the script filename from `sys.argv[0]`.

### Use Cases

- **Wrapped scripts** - When your script is called via a wrapper or alias
- **Clarity** - Set a clearer name than the actual filename
- **Documentation** - Use a name that matches your documentation

---

## Testing Help Generation

Access generated help text programmatically for testing and documentation:

```python
@proto.cli
def train(lr: float = 0.001):
  """Train a model."""
  pass

# Plain text (for testing, logs, pipes)
print(train.__help_str__)

# Colorized (for terminal display)
print(train.__ansi_str__)
```

### Use Cases

- **Unit tests** - Verify help text is generated correctly
- **Documentation generation** - Extract help text for docs
- **Debugging** - Inspect what users will see
- **CI/CD** - Validate help output in pipelines

### Example Test

```python
def test_help_text():
  @proto.cli
  def train(lr: float = 0.001):
    """Train a model."""
    pass

  help_text = train.__help_str__
  assert "--lr" in help_text
  assert "FLOAT" in help_text
  assert "default: 0.001" in help_text
```

---

## Edge Cases

### Long Parameter Names

Long parameter names are automatically wrapped in help text:

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

Numbers are preserved in parameter names and converted properly:

```python
@proto.cli
def train(
  model_2d: bool = False,      # → --model-2d
  resnet50_pretrained: bool = True,  # → --resnet50-pretrained
):
  pass
```

Numbers remain in both the parameter name and the CLI argument, making it easy to reference models by version or variant.

---

## Quick Pattern Reference

| Pattern | When to Use | Example |
|---------|------------|---------|
| **Basic CLI** | Simple scripts with few parameters | `@proto.cli def train(lr: float = 0.001)` |
| **Grouped Options** | Many related parameters | Use `@proto.prefix` classes for organization |
| **Custom Program Name** | Wrapped scripts or clarity | `@proto.cli(prog="my-tool")` |
| **Testing** | Verify help output | Access `__help_str__` attribute |

---

## Related

- [CLI Fundamentals](cli-fundamentals) - Basic CLI features
- [Naming Conventions](naming-conventions) - How names convert to CLI arguments
- [Help Generation](help-generation) - Documentation extraction
- [Advanced Patterns](advanced-patterns) - Prefixes and singletons
- [Type System](type-system) - Type handling and conversion
