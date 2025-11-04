# Quick Start Guide

## Installation

Install params-proto using pip:

```bash
pip install params-proto waterbear
```

## Basic Example

Here's a simple example to get you started with params-proto:

```python
# config.py
from params_proto.v2.proto import ParamsProto, Flag, Proto

class Config(ParamsProto):
    """Training configuration for my ML model"""
    
    # Boolean flags
    debug = Flag("Enable debug logging", default=False)
    use_gpu = Flag("Use GPU for training", default=True)
    
    # String parameters
    model_name = Proto("Model architecture", default="resnet50")
    dataset = Proto("Dataset to use", default="cifar10")
    
    # Numeric parameters
    learning_rate = Proto("Learning rate", default=0.001)
    batch_size = Proto("Batch size", default=32)
    epochs = Proto("Number of training epochs", default=100)
    
    # Path parameters with environment variable support
    data_dir = Proto("Data directory", default="${DATA_DIR:/tmp/data}")
    output_dir = Proto("Output directory", default="./outputs")

# train.py
from config import Config

def main():
    # Parse command line arguments
    Config.parse()
    
    print(f"Training {Config.model_name} on {Config.dataset}")
    print(f"Learning rate: {Config.learning_rate}")
    print(f"Batch size: {Config.batch_size}")
    print(f"Debug mode: {Config.debug}")

if __name__ == "__main__":
    main()
```

## Command Line Usage

Run your script with different parameters:

```bash
# Use default values
python train.py

# Override specific parameters
python train.py --Config.learning_rate 0.01 --Config.batch_size 64

# Enable debug mode
python train.py --Config.debug

# Use environment variables
DATA_DIR=/path/to/data python train.py

# Get help
python train.py --help
```

## Key Concepts

### ParamsProto Class
The base class for defining parameter schemas. Inherit from it to create your configuration classes.

### Proto Fields
Use `Proto()` to define parameters with type hints, default values, and help text.

### Flag Fields  
Use `Flag()` for boolean parameters that can be enabled with `--flag` or disabled with `--no-flag`.

### Environment Variables
Use `${VAR_NAME}` syntax in default values to read from environment variables. You can provide fallback values with `${VAR_NAME:fallback}`.

## IDE Support

params-proto provides excellent IDE support:

- **Autocompletion**: Access parameters with `Config.parameter_name`
- **Type hints**: Full type safety and checking
- **Documentation**: Hover over parameters to see help text
- **Refactoring**: Rename parameters safely across your codebase

## Next Steps

- Explore the [Examples](examples/index.md) for more advanced usage patterns
- Check the [API Reference](api/index.md) for detailed documentation
- Learn about nested configurations and parameter sweeps