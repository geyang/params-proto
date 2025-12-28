# ANSI Formatting

params-proto automatically generates colorized, terminal-aware help text for better readability.

## Overview

Help text comes in two flavors:

- **`__help_str__`**: Plain text (for testing, logs, pipes)
- **`__ansi_str__`**: Colorized with line wrapping (for terminal display)

## Color Scheme

When displayed in a terminal, help text is automatically colorized:

- **Type names** (INT, STR, FLOAT): **Bold Cyan**
- **(required)**: **Bold Red**
- **(default: ...)**: *Dim Cyan*
- **Option names** (--foo): **Bold**

## Example

```python
from params_proto import proto

@proto.cli
def train(
  data_path: str,  # Path to training data
  batch_size: int = 32,  # Number of samples per batch
):
  """Train a model."""
  pass

# Plain text (for testing)
print(train.__help_str__)

# Colorized (for terminal)
print(train.__ansi_str__)
```

Terminal output will show:
- `--data-path` and `--batch-size` in **bold**
- `STR` and `INT` in **bold cyan**
- `(required)` in **bold red**
- `(default: 32)` in *dim cyan*

## Line Wrapping

Long descriptions are automatically wrapped to fit terminal width:

```python
@proto.cli
def process(
  config: str,  # YAML configuration file specifying model architecture, hyperparameters, optimizer settings, and training schedule parameters
):
  """Process data."""
  pass
```

Output wraps intelligently:
```
--config STR         YAML configuration file specifying model
                     architecture, hyperparameters, optimizer settings,
                     and training schedule parameters (required)
```

## Terminal Width Detection

The module automatically detects terminal width using `shutil.get_terminal_size()`:

- **Default**: 80 characters (fallback)
- **Maximum**: 120 characters (for readability)
- **Adaptive**: Uses actual terminal width when available

## Disabling Colors

Colors are automatically disabled when:

- Output is piped: `python script.py --help | grep param`
- `NO_COLOR` environment variable is set
- `TERM=dumb`
- Output is not a TTY (terminal)

### Manual Control

```bash
# Disable colors
NO_COLOR=1 python script.py --help

# Force colors in pipes (with less -R for ANSI support)
FORCE_COLOR=1 python script.py --help | less -R
```

## Testing

Tests use `__help_str__` (plain text) to avoid brittle ANSI code assertions:

```python
def test_help_output():
  @proto.cli(prog='test')
  def func(param: int = 1):
    pass

  # Test plain text (stable, no ANSI codes)
  assert "--param INT" in func.__help_str__
  assert "(default: 1)" in func.__help_str__

  # ANSI version available but not tested
  ansi_help = func.__ansi_str__  # Has colors, wrapping
```

## Environment Variables

Standard environment variables are respected:

| Variable | Effect |
|----------|--------|
| `NO_COLOR` | Disable all ANSI colors |
| `FORCE_COLOR` | Force colors even when piped |
| `CLICOLOR=0` | Disable colors (standard) |
| `TERM=dumb` | Disable colors |

## Implementation Details

### Color Codes

```python
from params_proto.ansi_help import Colors

Colors.BOLD          # \033[1m
Colors.RED           # \033[31m
Colors.CYAN          # \033[36m
Colors.DIM           # \033[2m
Colors.RESET         # \033[0m
```

### Width Detection

```python
from params_proto.ansi_help import get_terminal_width

# Get current terminal width (default 80, max 120)
width = get_terminal_width()
```

### Colorization

```python
from params_proto.ansi_help import colorize_help

# Colorize and wrap plain help text
ansi_text = colorize_help(plain_help, width=100)
```

### Strip ANSI

```python
from params_proto.ansi_help import strip_ansi

# Remove ANSI codes for length calculations
plain = strip_ansi(colored_text)
```

## Best Practices

1. **Testing**: Always test `__help_str__`, not `__ansi_str__`
2. **Documentation**: Use inline comments for parameter descriptions
3. **Pipes**: Colors auto-disable for pipes and redirects
4. **Accessibility**: Respect `NO_COLOR` for screen readers
5. **Width**: Keep descriptions readable at 80 characters

## Customization

Currently, the color scheme is fixed for consistency. Future versions may support:

- Custom color schemes via config
- Per-parameter color overrides
- Rich text formatting (underline, backgrounds)
- Emoji support

See `ANSI_HELP_CONSIDERATIONS.md` in the repository for roadmap.

## Related

- [Configuration Basics](configuration_basics) - Functions and classes
- [CLI Generation](cli_generation) - How Python becomes CLI
- [Types Guide](types) - Type annotation support
- [Release Notes](../release_notes) - v3.0.0 ANSI features
