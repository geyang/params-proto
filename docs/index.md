# params-proto: Modern Hyper Parameter Management for Machine Learning

**params-proto** is a modern hyperparameter management library designed to solve "Experiment Parameter Hell" in machine
learning projects. 

## Quick Start

Install params-proto:

```shell
pip install params-proto
```
or
```shell
uv add params-proto
```

## Key Features

- **Declarative Parameter Definition**: Define parameters as Python classes for better IDE support
- **Command-line Integration**: Automatically generate CLI interfaces with argparse
- **Auto-completion**: Tab completion support for command-line arguments
- **Environment Variable Support**: Use environment variables as parameter defaults
- **Type Safety**: Full type hints and IDE support

## Basic Usage

```python
from params_proto import proto


@proto.prefix
class Args:
    """My experiment configuration"""

    debug: bool = False  # Enable debug mode
    model_name: str = "resnet50"  # Model architecture to use
    learning_rate: float = 0.001  # Learning rate for training
    data_path: str = "${DATA_PATH}"  # Path to training data (supports env vars)
```

Use from the command-line:

```bash
python train.py --Args.debug --Args.learning_rate 0.01 --Args.model_name "transformer"
```

## Documentation Sections

```{toctree}
:maxdepth: 2
:caption: User Guide

quick_start
release_notes
examples/index
api/index
```

## What Problem Does This Solve?

"Experiment Parameter Hell" occurs when you have numerous parameters scattered across your ML codebase, making it hard
to:

- Track what parameters exist
- Get IDE autocompletion and type checking
- Change parameters from the command line
- Maintain parameter documentation

params-proto solves this by providing a declarative way to define parameters that integrates seamlessly with Python IDEs
and command-line interfaces.

## GitHub Repository

The source code is available on [GitHub](https://github.com/geyang/params-proto).

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`