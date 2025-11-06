# Advanced Features

This tutorial covers advanced features of params-proto including nested configurations, environment variables, and custom validation.

## Environment Variables

Proto supports reading default values from environment variables, making your configurations flexible across different deployment environments.

### Basic Environment Variable Usage

```python
from params_proto.v2.proto import ParamsProto, Proto
import os

class Config(ParamsProto):
    # Read from environment variable with fallback
    database_url = Proto("Database URL", env="DATABASE_URL", default="sqlite:///default.db")
    
    # Required environment variable (strict parsing)
    api_key = Proto("API key", env="API_KEY", strict_parsing=True)
    
    # Environment variable with type conversion
    port = Proto("Server port", env="PORT", dtype=int, default=8000)
    
    # Variable expansion syntax
    log_file = Proto("Log file path", default="${HOME}/app.log")
    data_dir = Proto("Data directory", default="${DATA_DIR:/tmp/data}")

# Set environment variables
os.environ['DATABASE_URL'] = 'postgresql://localhost/myapp'
os.environ['API_KEY'] = 'secret-key-123'
os.environ['PORT'] = '9000'

Config.parse()

print(f"Database: {Config.database_url}")  # postgresql://localhost/myapp
print(f"API Key: {Config.api_key}")        # secret-key-123  
print(f"Port: {Config.port}")              # 9000 (converted to int)
print(f"Log file: {Config.log_file}")      # /Users/username/app.log
print(f"Data dir: {Config.data_dir}")      # /tmp/data (fallback used)
```

### Environment Variable Patterns

```python
class DeploymentConfig(ParamsProto):
    # Service configuration
    service_name = Proto("Service name", env="SERVICE_NAME", default="myapp")
    environment = Proto("Environment", env="ENVIRONMENT", default="development")
    
    # Database configuration
    db_host = Proto("Database host", env="DB_HOST", default="localhost")
    db_port = Proto("Database port", env="DB_PORT", dtype=int, default=5432)
    db_name = Proto("Database name", env="DB_NAME", default="myapp")
    
    # With complex fallbacks
    redis_url = Proto("Redis URL", default="${REDIS_URL:redis://localhost:6379}")
    
    # Boolean from environment
    debug = Proto("Debug mode", env="DEBUG", dtype=bool, default=False)

# Usage in different environments:
# Development: Uses defaults
# Staging: export ENVIRONMENT=staging DB_HOST=staging-db.example.com
# Production: export ENVIRONMENT=production DB_HOST=prod-db.example.com DEBUG=false
```

## Nested Configurations

Create hierarchical parameter structures for complex applications.

### Basic Nested Configuration

```python
from params_proto.v2.proto import ParamsProto, Proto

class DatabaseConfig(ParamsProto, cli=False):
    host = Proto("Database host", default="localhost")
    port = Proto("Database port", default=5432)
    name = Proto("Database name", default="myapp")
    user = Proto("Database user", default="postgres")

class RedisConfig(ParamsProto, cli=False):
    host = Proto("Redis host", default="localhost")
    port = Proto("Redis port", default=6379)
    db = Proto("Redis database", default=0)

class Config(ParamsProto):
    # Nested configurations
    database = DatabaseConfig
    redis = RedisConfig
    
    # Top-level configuration
    app_name = Proto("Application name", default="MyApp")
    debug = Proto("Debug mode", default=False)

# Parse command line arguments
Config.parse()

# Access nested parameters
print(f"App: {Config.app_name}")
print(f"Database: {Config.database.host}:{Config.database.port}/{Config.database.name}")
print(f"Redis: {Config.redis.host}:{Config.redis.port}/{Config.redis.db}")
```

### Command Line Usage with Nested Configs

```bash
# Override nested parameters
python app.py \
    --Config.app_name "Production App" \
    --DatabaseConfig.host "prod-db.example.com" \
    --DatabaseConfig.port 5433 \
    --RedisConfig.host "redis.example.com"
```

### Prefix-Based Nested Configuration

```python
from params_proto.v2.proto import ParamsProto, Proto, PrefixProto

class ModelConfig(PrefixProto, prefix="model"):
    """Model configuration with automatic prefix"""
    architecture = Proto("Model architecture", default="resnet50")
    layers = Proto("Number of layers", default=18)
    dropout = Proto("Dropout rate", default=0.1)

class TrainingConfig(PrefixProto, prefix="training"):
    """Training configuration with automatic prefix"""
    learning_rate = Proto("Learning rate", default=0.001)
    batch_size = Proto("Batch size", default=32)
    epochs = Proto("Number of epochs", default=100)

class Config(ParamsProto):
    experiment_name = Proto("Experiment name", default="baseline")
    seed = Proto("Random seed", default=42)

# All configs are parsed together
ModelConfig.parse()
TrainingConfig.parse()  
Config.parse()

# Command line usage:
# python train.py \
#     --model.architecture "transformer" \
#     --model.layers 12 \
#     --training.learning_rate 0.0001 \
#     --Config.experiment_name "transformer_baseline"
```

## Dynamic Configuration

Create configurations that adapt based on other parameter values.

### Conditional Configuration

```python
from params_proto.v2.proto import ParamsProto, Proto

class Config(ParamsProto, cli_parse=False):
    """Configuration with conditional logic"""
    
    # Base parameters
    model_type = Proto("Model type", default="cnn")
    dataset = Proto("Dataset", default="cifar10")
    
    def __init__(self, _deps=None):
        """Initialize with conditional logic"""
        super().__init__(_deps)
        
        # Set parameters based on model type
        if self.model_type == "cnn":
            self.hidden_size = 128
            self.num_layers = 3
        elif self.model_type == "transformer":
            self.hidden_size = 512
            self.num_layers = 6
            self.num_heads = 8
        
        # Set parameters based on dataset
        if self.dataset == "cifar10":
            self.num_classes = 10
            self.image_size = 32
        elif self.dataset == "imagenet":
            self.num_classes = 1000
            self.image_size = 224

# Parse and create instance
Config.parse()
config = Config()

print(f"Model: {config.model_type}")
print(f"Hidden size: {config.hidden_size}")
print(f"Classes: {config.num_classes}")

# Can override via command line:
# python train.py --Config.model_type transformer --Config.dataset imagenet
```

### Multi-Stage Configuration

```python
class Root(ParamsProto, cli_parse=False):
    """Root configuration that affects others"""
    launch_type = Proto("Launch type", default="local")
    environment = Proto("Environment", default="development")

class Config(ParamsProto):
    """Main configuration that depends on Root"""
    
    # Base parameters
    batch_size = Proto("Batch size", default=32)
    
    def __init__(self, _deps=None):
        # Initialize root configuration first
        root = Root(_deps)
        
        # Set parameters based on launch type
        if root.launch_type == "cluster":
            self.num_workers = 8
            self.gpu_count = 4
        else:
            self.num_workers = 2
            self.gpu_count = 1
            
        # Set parameters based on environment
        if root.environment == "production":
            self.log_level = "INFO"
            self.debug = False
        else:
            self.log_level = "DEBUG" 
            self.debug = True

# Usage
config = Config()
print(f"Workers: {config.num_workers}")
print(f"Debug: {config.debug}")

# Override root config affects child config
config = Config({"Root.launch_type": "cluster", "Root.environment": "production"})
print(f"Workers: {config.num_workers}")  # 8
print(f"Debug: {config.debug}")          # False
```

## Custom Validation

Add validation logic to ensure parameter values are correct.

### Basic Validation

```python
from params_proto.v2.proto import ParamsProto, Proto

class Config(ParamsProto):
    learning_rate = Proto("Learning rate", default=0.001)
    batch_size = Proto("Batch size", default=32) 
    epochs = Proto("Number of epochs", default=100)
    model_name = Proto("Model name", default="resnet50")
    
    @classmethod
    def validate(cls):
        """Validate parameter values"""
        # Validate ranges
        assert 0 < cls.learning_rate < 1, f"Learning rate must be between 0 and 1, got {cls.learning_rate}"
        assert cls.batch_size > 0, f"Batch size must be positive, got {cls.batch_size}"
        assert cls.epochs > 0, f"Epochs must be positive, got {cls.epochs}"
        
        # Validate choices
        valid_models = ["resnet50", "resnet101", "transformer", "vit"]
        assert cls.model_name in valid_models, f"Model must be one of {valid_models}, got {cls.model_name}"
        
        # Validate combinations
        if cls.model_name == "transformer" and cls.batch_size < 16:
            raise ValueError("Transformer models require batch_size >= 16")

# Parse and validate
Config.parse()
try:
    Config.validate()
    print("Configuration is valid!")
except AssertionError as e:
    print(f"Configuration error: {e}")
```

### Complex Validation

```python
class AdvancedConfig(ParamsProto):
    # Model parameters
    model_type = Proto("Model type", default="cnn")
    input_size = Proto("Input size", default=224)
    
    # Training parameters  
    optimizer = Proto("Optimizer", default="adam")
    learning_rate = Proto("Learning rate", default=0.001)
    weight_decay = Proto("Weight decay", default=1e-4)
    
    # Data parameters
    dataset = Proto("Dataset", default="imagenet")
    augmentation = Proto("Data augmentation", default=True)
    
    @classmethod
    def validate(cls):
        """Complex validation logic"""
        
        # Model-specific validation
        if cls.model_type == "vit":
            # Vision Transformer requires specific input sizes
            valid_sizes = [224, 256, 384, 512]
            assert cls.input_size in valid_sizes, \
                f"ViT requires input_size in {valid_sizes}, got {cls.input_size}"
        
        # Optimizer-specific validation
        if cls.optimizer == "sgd":
            assert cls.learning_rate >= 0.01, \
                "SGD typically requires learning_rate >= 0.01"
        elif cls.optimizer == "adam":
            assert cls.learning_rate <= 0.01, \
                "Adam typically works best with learning_rate <= 0.01"
        
        # Dataset-specific validation
        if cls.dataset == "cifar10":
            assert cls.input_size <= 64, \
                "CIFAR-10 images are 32x32, input_size should be <= 64"
        elif cls.dataset == "imagenet":
            assert cls.input_size >= 224, \
                "ImageNet typically requires input_size >= 224"
        
        # Cross-parameter validation
        if cls.weight_decay > 1e-2 and cls.optimizer == "adam":
            print("Warning: High weight_decay with Adam might cause issues")

# Usage with validation
try:
    AdvancedConfig.parse()
    AdvancedConfig.validate()
    print("✓ All validations passed")
except (AssertionError, ValueError) as e:
    print(f"✗ Validation failed: {e}")
```

## Configuration Updates

Dynamically update configurations during runtime.

### Runtime Updates

```python
class Config(ParamsProto):
    learning_rate = Proto("Learning rate", default=0.001)
    batch_size = Proto("Batch size", default=32)
    model_name = Proto("Model name", default="resnet50")

# Initial configuration
Config.parse()
print(f"Initial LR: {Config.learning_rate}")

# Update single parameter
Config._update(learning_rate=0.01)
print(f"Updated LR: {Config.learning_rate}")

# Update multiple parameters
Config._update(learning_rate=0.001, batch_size=64, model_name="transformer")
print(f"Batch size: {Config.batch_size}")
print(f"Model: {Config.model_name}")

# Update from dictionary
updates = {
    "learning_rate": 0.0001,
    "batch_size": 128
}
Config._update(updates)
print(f"Dict update LR: {Config.learning_rate}")
```

### Scoped Updates

```python
from contextlib import contextmanager

class Config(ParamsProto):
    learning_rate = Proto(default=0.001)
    batch_size = Proto(default=32)

@contextmanager
def config_scope(**updates):
    """Temporarily update configuration"""
    # Save current values
    original = {k: getattr(Config, k) for k in updates.keys()}
    
    try:
        # Apply updates
        Config._update(**updates)
        yield Config
    finally:
        # Restore original values
        Config._update(**original)

# Normal configuration
print(f"Normal LR: {Config.learning_rate}")  # 0.001

# Temporarily updated configuration
with config_scope(learning_rate=0.01, batch_size=64):
    print(f"Scoped LR: {Config.learning_rate}")      # 0.01
    print(f"Scoped batch: {Config.batch_size}")      # 64

# Back to original
print(f"Restored LR: {Config.learning_rate}")       # 0.001
print(f"Restored batch: {Config.batch_size}")       # 32
```

## Integration Patterns

Common patterns for integrating params-proto with other libraries.

### Logging Configuration

```python
import logging
from params_proto.v2.proto import ParamsProto, Proto, Flag

class LoggingConfig(ParamsProto):
    log_level = Proto("Log level", default="INFO")
    log_file = Proto("Log file", default="app.log")
    log_format = Proto("Log format", default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_output = Flag("Output to console", default=True)

def setup_logging():
    """Configure logging based on params"""
    LoggingConfig.parse()
    
    # Configure root logger
    level = getattr(logging, LoggingConfig.log_level.upper())
    logging.basicConfig(
        level=level,
        format=LoggingConfig.log_format,
        filename=LoggingConfig.log_file
    )
    
    # Add console handler if requested
    if LoggingConfig.console_output:
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter(LoggingConfig.log_format)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

# Usage
setup_logging()
logger = logging.getLogger(__name__)
logger.info("Application started")
```

### Framework Integration

```python
# PyTorch integration example
import torch
import torch.nn as nn
import torch.optim as optim

class TrainingConfig(ParamsProto):
    # Model parameters
    model_type = Proto("Model type", default="resnet")
    num_classes = Proto("Number of classes", default=10)
    
    # Training parameters
    learning_rate = Proto("Learning rate", default=0.001)
    weight_decay = Proto("Weight decay", default=1e-4)
    optimizer_type = Proto("Optimizer", default="adam")
    scheduler_type = Proto("Scheduler", default="cosine")
    
    # Hardware
    device = Proto("Device", default="auto")
    mixed_precision = Flag("Mixed precision training", default=False)

def create_model():
    """Create model based on configuration"""
    TrainingConfig.parse()
    
    if TrainingConfig.model_type == "resnet":
        import torchvision.models as models
        model = models.resnet50(num_classes=TrainingConfig.num_classes)
    else:
        raise ValueError(f"Unknown model type: {TrainingConfig.model_type}")
    
    # Move to device
    if TrainingConfig.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(TrainingConfig.device)
    
    model = model.to(device)
    return model, device

def create_optimizer(model):
    """Create optimizer based on configuration"""
    if TrainingConfig.optimizer_type == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=TrainingConfig.learning_rate,
            weight_decay=TrainingConfig.weight_decay
        )
    elif TrainingConfig.optimizer_type == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=TrainingConfig.learning_rate,
            weight_decay=TrainingConfig.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {TrainingConfig.optimizer_type}")
    
    return optimizer

# Usage
model, device = create_model()
optimizer = create_optimizer(model)
print(f"Model: {TrainingConfig.model_type}")
print(f"Device: {device}")
print(f"Optimizer: {TrainingConfig.optimizer_type}")
```