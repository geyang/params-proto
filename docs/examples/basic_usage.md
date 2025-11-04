# Basic Usage Examples

## Simple Configuration

Here's a basic example showing how to define and use a configuration class:

```python
from params_proto.v2.proto import ParamsProto, Flag, Proto

class TrainingConfig(ParamsProto):
    """Configuration for model training"""
    
    # Model parameters
    model_type = Proto("Type of model to train", default="cnn")
    hidden_size = Proto("Hidden layer size", default=128)
    
    # Training parameters
    learning_rate = Proto("Learning rate", default=0.001)
    batch_size = Proto("Batch size", default=32)
    epochs = Proto("Number of epochs", default=10)
    
    # Flags
    use_gpu = Flag("Use GPU for training", default=True)
    save_checkpoints = Flag("Save model checkpoints", default=True)

def train_model():
    TrainingConfig.parse()
    
    print(f"Training {TrainingConfig.model_type} model")
    print(f"Hidden size: {TrainingConfig.hidden_size}")
    print(f"Learning rate: {TrainingConfig.learning_rate}")
    print(f"Batch size: {TrainingConfig.batch_size}")
    print(f"Epochs: {TrainingConfig.epochs}")
    print(f"Using GPU: {TrainingConfig.use_gpu}")
    print(f"Save checkpoints: {TrainingConfig.save_checkpoints}")

if __name__ == "__main__":
    train_model()
```

### Command Line Usage

```bash
# Use default values
python train.py

# Override parameters
python train.py --TrainingConfig.learning_rate 0.01 --TrainingConfig.batch_size 64

# Enable/disable flags
python train.py --TrainingConfig.use_gpu --no-TrainingConfig.save_checkpoints

# Get help
python train.py --help
```

## Data Loading Configuration

```python
from params_proto.v2.proto import ParamsProto, Proto, Flag

class DataConfig(ParamsProto):
    """Data loading and preprocessing configuration"""
    
    # Data paths
    train_path = Proto("Path to training data", default="./data/train")
    val_path = Proto("Path to validation data", default="./data/val")
    test_path = Proto("Path to test data", default="./data/test")
    
    # Data processing
    image_size = Proto("Input image size", default=224)
    num_workers = Proto("Number of data loader workers", default=4)
    shuffle = Flag("Shuffle training data", default=True)
    augment = Flag("Apply data augmentation", default=True)

def load_data():
    DataConfig.parse()
    
    print(f"Loading data from:")
    print(f"  Train: {DataConfig.train_path}")
    print(f"  Val: {DataConfig.val_path}")
    print(f"  Test: {DataConfig.test_path}")
    print(f"Image size: {DataConfig.image_size}")
    print(f"Workers: {DataConfig.num_workers}")
    print(f"Shuffle: {DataConfig.shuffle}")
    print(f"Augment: {DataConfig.augment}")

if __name__ == "__main__":
    load_data()
```

## Multiple Configuration Classes

You can use multiple configuration classes in the same script:

```python
from params_proto.v2.proto import ParamsProto, Proto, Flag

class ModelConfig(ParamsProto):
    """Model architecture configuration"""
    architecture = Proto("Model architecture", default="resnet")
    layers = Proto("Number of layers", default=18)
    dropout = Proto("Dropout rate", default=0.1)

class OptimConfig(ParamsProto):
    """Optimizer configuration"""
    optimizer = Proto("Optimizer type", default="adam")
    learning_rate = Proto("Learning rate", default=0.001)
    weight_decay = Proto("Weight decay", default=1e-4)

class ExperimentConfig(ParamsProto):
    """Experiment configuration"""
    experiment_name = Proto("Name of experiment", default="baseline")
    seed = Proto("Random seed", default=42)
    log_dir = Proto("Logging directory", default="./logs")

def main():
    # Parse all configurations
    ModelConfig.parse()
    OptimConfig.parse()
    ExperimentConfig.parse()
    
    print("=== Model Configuration ===")
    print(f"Architecture: {ModelConfig.architecture}")
    print(f"Layers: {ModelConfig.layers}")
    print(f"Dropout: {ModelConfig.dropout}")
    
    print("\n=== Optimizer Configuration ===")
    print(f"Optimizer: {OptimConfig.optimizer}")
    print(f"Learning rate: {OptimConfig.learning_rate}")
    print(f"Weight decay: {OptimConfig.weight_decay}")
    
    print("\n=== Experiment Configuration ===")
    print(f"Experiment: {ExperimentConfig.experiment_name}")
    print(f"Seed: {ExperimentConfig.seed}")
    print(f"Log dir: {ExperimentConfig.log_dir}")

if __name__ == "__main__":
    main()
```

### Usage with Multiple Configs

```bash
python experiment.py \
    --ModelConfig.architecture "transformer" \
    --ModelConfig.layers 12 \
    --OptimConfig.learning_rate 0.0001 \
    --ExperimentConfig.experiment_name "transformer_baseline"
```