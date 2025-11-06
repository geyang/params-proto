# Utility Functions API Reference

Helper functions for working with params-proto configurations.

## `dot_to_deps()`

Convert dot-notation dictionaries to nested structures.

```python
from params_proto.utils import dot_to_deps

# Flat dictionary with dot notation
flat = {
    "Model.name": "resnet50",
    "Model.pretrained": True,
    "Training.lr": 0.001,
    "Training.batch_size": 32,
}

# Convert to nested structure
nested = dot_to_deps(flat)
# {
#     "Model": {"name": "resnet50", "pretrained": True},
#     "Training": {"lr": 0.001, "batch_size": 32}
# }
```

**Parameters:**
- `deps` (dict) - Flat dictionary with dot-notation keys

**Returns:**
- `dict` - Nested dictionary structure

**Usage:**
- Converting CLI arguments to nested configs
- Processing configuration files
- Preparing arguments for `proto.bind()`

## `parse_env_template()`

Extract environment variable names from template strings.

```python
from params_proto.parse_env_template import parse_env_template

template = "Server: $HOST:$PORT, DB: $DATABASE_URL"
variables = parse_env_template(template)
# ["HOST", "PORT", "DATABASE_URL"]
```

**Parameters:**
- `template` (str) - String containing `$VARIABLE_NAME` patterns

**Returns:**
- `list[str]` - List of environment variable names found

**Usage:**
- Validating required environment variables
- Extracting variable dependencies from configs
- Generating documentation

## `all_available()`

Check if all environment variables in a template are available.

```python
from params_proto.parse_env_template import all_available
import os

# Set some environment variables
os.environ["HOST"] = "localhost"
os.environ["PORT"] = "8000"

template = "Server: $HOST:$PORT"
is_ready = all_available(template, os.environ)  # True

template2 = "DB: $MISSING_VAR"
is_ready2 = all_available(template2, os.environ)  # False
```

**Parameters:**
- `template` (str) - String containing `$VARIABLE_NAME` patterns
- `env` (dict) - Environment dictionary (usually `os.environ`)
- `strict` (bool, optional) - If True, empty values count as unavailable

**Returns:**
- `bool` - True if all variables are available and non-empty

**Usage:**
- Validating configuration before startup
- Environment readiness checks
- Pre-flight validation

## Example: Environment Variable Configs

```python
import os
from params_proto import proto
from params_proto.parse_env_template import parse_env_template, all_available

@proto
class DatabaseConfig:
    """Database configuration."""

    # Use environment variables with fallbacks
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "mydb")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "")

    @property
    def url(self) -> str:
        """Build database URL from components."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

# Validate environment
required_template = "DB: $DB_HOST:$DB_PORT/$DB_NAME (user: $DB_USER)"
if not all_available(required_template, os.environ):
    missing = [v for v in parse_env_template(required_template)
               if v not in os.environ]
    print(f"Missing environment variables: {missing}")
else:
    config = DatabaseConfig()
    print(f"Database URL: {config.url}")
```

## Example: Configuration Merging

```python
from params_proto import proto
from params_proto.utils import dot_to_deps

# Load from command line (flat format)
cli_args = {
    "Model.name": "vit",
    "Model.pretrained": True,
    "Training.lr": 0.0001,
    "Training.epochs": 50,
}

# Convert to nested format
nested_config = dot_to_deps(cli_args)

# Apply to proto configs
proto.bind(**cli_args)  # Simpler approach

# Or manually update singletons
for prefix, values in nested_config.items():
    # Access singleton and update
    # (requires access to internal _SINGLETONS)
    pass
```

## See Also

- [Quick Start](../quick_start.md) - Basic usage examples
- [API Reference](proto.md) - Core proto module
