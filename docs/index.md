# params-proto: Modern Hyper Parameter Management for Machine Learning

Welcome to the documentation for **params-proto**, a modern hyperparameter management library designed to solve "Experiment Parameter Hell" in machine learning projects.

## Quick Start

Install params-proto:

```bash
pip install params-proto waterbear
```

## Key Features

- **Declarative Parameter Definition**: Define parameters as Python classes for better IDE support
- **Command-line Integration**: Automatically generate CLI interfaces with argparse
- **Auto-completion**: Tab completion support for command-line arguments
- **Environment Variable Support**: Use environment variables as parameter defaults
- **Type Safety**: Full type hints and IDE support

## Basic Usage

```python
from params_proto.proto import ParamsProto, Flag, Proto

class Args(ParamsProto):
    """My experiment configuration"""
    
    # Boolean flag with help text
    debug = Flag("Enable debug mode", default=False)
    
    # String parameter with default
    model_name = Proto("Model architecture to use", default="resnet50")
    
    # Numeric parameter
    learning_rate = Proto("Learning rate for training", default=0.001)
    
    # Environment variable support
    data_path = Proto("Path to training data", default="${DATA_PATH}", dtype=str)
```

Use from command line:
```bash
python train.py --Args.debug --Args.learning_rate 0.01 --Args.model_name "transformer"
```

## Documentation Sections

```{toctree}
:maxdepth: 2
:caption: User Guide

quick_start
examples/index
api/index
```

## What Problem Does This Solve?

"Experiment Parameter Hell" occurs when you have numerous parameters scattered across your ML codebase, making it hard to:

- Track what parameters exist
- Get IDE autocompletion and type checking
- Change parameters from the command line
- Maintain parameter documentation

params-proto solves this by providing a declarative way to define parameters that integrates seamlessly with Python IDEs and command-line interfaces.

## GitHub Repository

The source code is available on [GitHub](https://github.com/episodeyang/params_proto).

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`