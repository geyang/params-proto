# proto module

The `proto` module contains the core classes for defining and managing parameters in params-proto.

## Core Classes

### ParamsProto

The main base class for creating parameter configurations. Inherit from this class to define your parameter schemas.

```{eval-rst}
.. autoclass:: params_proto.v2.ParamsProto
   :members:
   :undoc-members:
   :show-inheritance:
```

#### Key Methods

- **`parse()`**: Parse command-line arguments and update parameter values
- **`to_dict()`**: Convert the configuration to a dictionary
- **`_update()`**: Update parameter values programmatically
- **`__vars__`**: Property that returns nested dictionary representation

#### Usage Example

```python
from params_proto.v2.proto import ParamsProto, Proto, Flag

class Config(ParamsProto):
    learning_rate = Proto("Learning rate", default=0.001)
    debug = Flag("Enable debug mode", default=False)

# Parse command line arguments
Config.parse()

# Access parameters
print(Config.learning_rate)  # 0.001
print(Config.debug)         # False
```

### Proto

Defines a parameter with type, default value, help text, and environment variable support.

```{eval-rst}
.. autoclass:: params_proto.v2.proto.Proto
   :members:
   :undoc-members:
   :show-inheritance:
```

#### Parameters

- **`default`**: Default value for the parameter
- **`help`**: Help text displayed in CLI
- **`dtype`**: Type of the parameter (inferred from default if not specified)
- **`env`**: Environment variable name to read default from
- **`strict_parsing`**: Whether to raise error for unset environment variables

#### Usage Examples

```python
# Basic parameter
name = Proto("Model name", default="resnet50")

# With type specification
epochs = Proto("Training epochs", default=100, dtype=int)

# With environment variable
data_path = Proto("Data directory", default="${DATA_DIR:/tmp/data}")

# With strict environment parsing
api_key = Proto("API key", env="API_KEY", strict_parsing=True)
```

### Flag

A boolean parameter that can be enabled/disabled from command line.

```{eval-rst}
.. autoclass:: params_proto.v2.proto.Flag
   :members:
   :undoc-members:
   :show-inheritance:
```

#### Parameters

- **`help`**: Help text for the flag
- **`to_value`**: Value when flag is enabled (default: True)
- **`default`**: Default value when flag is not specified

#### Usage Examples

```python
# Basic flag
debug = Flag("Enable debug logging")

# Custom flag value
verbose = Flag("Verbose output", to_value=2, default=0)

# Usage from command line:
# --Config.debug        # Sets debug=True
# --no-Config.debug     # Sets debug=False
# --Config.verbose      # Sets verbose=2
```

## Utility Classes

### PrefixProto

A ParamsProto with automatic prefix generation for nested configurations.

```{eval-rst}
.. autoclass:: params_proto.v2.proto.PrefixProto
   :members:
   :show-inheritance:
```

### Meta

The metaclass that powers ParamsProto functionality.

```{eval-rst}
.. autoclass:: params_proto.v2.proto.Meta
   :members:
   :show-inheritance:
```

## Advanced Features

### Environment Variable Support

Proto supports reading default values from environment variables:

```python
# Using environment variables
database_url = Proto("Database URL", env="DATABASE_URL")
debug_level = Proto("Debug level", env="DEBUG_LEVEL", dtype=int, default=0)

# With variable expansion
log_dir = Proto("Log directory", default="${HOME}/logs")
```

### Nested Configurations

Create hierarchical parameter structures:

```python
class DatabaseConfig(ParamsProto):
    host = Proto("Database host", default="localhost")
    port = Proto("Database port", default=5432)

class Config(ParamsProto):
    database = DatabaseConfig
    app_name = Proto("Application name", default="myapp")
```

### Custom Validation

Override methods to add custom validation:

```python
class Config(ParamsProto):
    learning_rate = Proto("Learning rate", default=0.001)
    
    @classmethod
    def validate(cls):
        assert 0 < cls.learning_rate < 1, "Learning rate must be between 0 and 1"
```

## Complete Module Reference

```{eval-rst}
.. automodule:: params_proto.v2.proto
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: ParamsProto, Proto, Flag, PrefixProto, Meta, Bear, SimpleNamespace, ChainMap, defaultdict, suppress, isfunction, ismethod, BuiltinFunctionType, chain
   :imported-members: False
```