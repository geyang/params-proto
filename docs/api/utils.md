# utils module

The `utils` module provides utility functions for configuration management, ANSI color handling, and YAML processing.

## Core Functions

### Configuration Management

#### dot_to_deps

Convert dot-separated dictionary keys into nested dependencies.

```{eval-rst}
.. autofunction:: params_proto.v2.utils.dot_to_deps
```

**Usage Example:**

```python
from params_proto.v2.utils import dot_to_deps

# Convert dot notation to nested structure
dot_dict = {
    "model.learning_rate": 0.001,
    "model.batch_size": 32,
    "data.path": "/tmp/data",
    "data.augment": True
}

deps = dot_to_deps(dot_dict, "model")
# Results in: {'learning_rate': 0.001, 'batch_size': 32}

data_deps = dot_to_deps(dot_dict, "data")
# Results in: {'path': '/tmp/data', 'augment': True}
```

#### flatten

Flatten nested dictionaries into dot-separated keys.

```{eval-rst}
.. autofunction:: params_proto.v2.utils.flatten
```

**Usage Example:**

```python
from params_proto.v2.utils import flatten

nested_config = {
    "model": {
        "learning_rate": 0.001,
        "architecture": {
            "layers": 3,
            "hidden_size": 128
        }
    },
    "training": {
        "epochs": 100,
        "batch_size": 32
    }
}

flat_config = flatten(nested_config)
# Results in:
# {
#     "model.learning_rate": 0.001,
#     "model.architecture.layers": 3,
#     "model.architecture.hidden_size": 128,
#     "training.epochs": 100,
#     "training.batch_size": 32
# }
```

### YAML Configuration Loading

#### read_deps

Read and process YAML configuration files with template inheritance.

```{eval-rst}
.. autofunction:: params_proto.v2.utils.read_deps
```

**Features:**
- Automatic flattening of nested configurations
- Base template inheritance via `_base` key
- Environment variable expansion

**Usage Example:**

```python
from params_proto.v2.utils import read_deps

# Load configuration from YAML file
config = read_deps("configs/experiment.yaml")

# The function automatically:
# 1. Loads the YAML file
# 2. Processes any _base template inheritance
# 3. Flattens nested dictionaries
# 4. Expands environment variables
```

**YAML File Example:**

```yaml
# configs/base.yaml
model:
  learning_rate: 0.001
  batch_size: 32

training:
  epochs: 100
  optimizer: adam

# configs/experiment.yaml  
_base: base.yaml

model:
  learning_rate: 0.01  # Override base value
  
data:
  path: ${DATA_PATH:/tmp/data}  # Environment variable with fallback
  workers: 4
```

### Text Processing

#### clean_ansi

Remove ANSI color codes from strings.

```{eval-rst}
.. autofunction:: params_proto.v2.utils.clean_ansi
```

**Usage Example:**

```python
from params_proto.v2.utils import clean_ansi

# Remove color codes from terminal output
colored_text = "\x1b[31mError:\x1b[0m Something went wrong"
clean_text = clean_ansi(colored_text)
# Results in: "Error: Something went wrong"
```

## Advanced Configuration Patterns

### Hierarchical Configurations

Use `dot_to_deps` to create hierarchical parameter structures:

```python
from params_proto.v2.proto import ParamsProto, Proto
from params_proto.v2.utils import dot_to_deps

class ModelConfig(ParamsProto):
    learning_rate = Proto(default=0.001)
    batch_size = Proto(default=32)

class DataConfig(ParamsProto):
    path = Proto(default="/tmp/data")
    workers = Proto(default=4)

# Load from flat configuration
config_dict = {
    "model.learning_rate": 0.01,
    "model.batch_size": 64,
    "data.path": "/data/train",
    "data.workers": 8
}

# Update configurations
model_deps = dot_to_deps(config_dict, "model")
ModelConfig._update(model_deps)

data_deps = dot_to_deps(config_dict, "data")
DataConfig._update(data_deps)
```

### Configuration Templates

Create reusable configuration templates:

```yaml
# templates/base_model.yaml
model:
  architecture: resnet50
  dropout: 0.1
  batch_norm: true

training:
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 1e-4

# experiments/vision_experiment.yaml
_base: ../templates/base_model.yaml

model:
  architecture: resnet101  # Override base
  input_size: 224

training:
  learning_rate: 0.0001    # Override base
  epochs: 200

data:
  augmentation: true
  normalize: true
```

### Environment-Based Configs

Use environment variables for deployment flexibility:

```yaml
# config.yaml
database:
  url: ${DATABASE_URL:sqlite:///default.db}
  pool_size: ${DB_POOL_SIZE:5}

api:
  key: ${API_KEY}  # Required environment variable
  timeout: ${API_TIMEOUT:30}

logging:
  level: ${LOG_LEVEL:INFO}
  file: ${LOG_FILE:/var/log/app.log}
```

```python
# In your application
import os
from params_proto.v2.utils import read_deps

# Set environment variables
os.environ['DATABASE_URL'] = 'postgresql://user:pass@localhost/db'
os.environ['API_KEY'] = 'secret-key-123'
os.environ['LOG_LEVEL'] = 'DEBUG'

# Load configuration with environment expansion
config = read_deps("config.yaml")
```

### Selective Flattening

Control which parts of the configuration get flattened:

```yaml
# config.yaml
model:
  learning_rate: 0.001
  architecture:
    _flatten: false  # Keep this nested
    layers: [64, 128, 256]
    activation: relu
    
training:
  epochs: 100
  batch_size: 32
```

```python
from params_proto.v2.utils import read_deps

config = read_deps("config.yaml")
# Results in:
# {
#     "model.learning_rate": 0.001,
#     "model.architecture": {  # Not flattened due to _flatten: false
#         "layers": [64, 128, 256],
#         "activation": "relu"
#     },
#     "training.epochs": 100,
#     "training.batch_size": 32
# }
```

## Complete Module Reference

```{eval-rst}
.. automodule:: params_proto.v2.utils
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: dot_to_deps, flatten, read_deps, clean_ansi
```