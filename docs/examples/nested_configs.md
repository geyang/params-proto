# Nested Configurations

Learn how to create hierarchical parameter structures for complex applications using params-proto's nested configuration features.

## Basic Nested Configurations

Nested configurations help organize parameters into logical groups, making large configurations more manageable.

### Simple Nested Structure

```python
from params_proto.v2.proto import ParamsProto, Proto

class DatabaseConfig(ParamsProto, cli=False):
    """Database connection configuration"""
    host = Proto("Database host", default="localhost")
    port = Proto("Database port", default=5432)
    name = Proto("Database name", default="myapp")
    user = Proto("Database user", default="postgres")
    password = Proto("Database password", default="password")
    pool_size = Proto("Connection pool size", default=5)

class RedisConfig(ParamsProto, cli=False):
    """Redis cache configuration"""
    host = Proto("Redis host", default="localhost")  
    port = Proto("Redis port", default=6379)
    db = Proto("Redis database number", default=0)
    password = Proto("Redis password", default=None)

class Config(ParamsProto):
    """Main application configuration"""
    
    # Nested configurations
    database = DatabaseConfig
    redis = RedisConfig
    
    # Top-level parameters
    app_name = Proto("Application name", default="MyApp")
    debug = Proto("Debug mode", default=False)
    log_level = Proto("Logging level", default="INFO")

# Parse all configurations
Config.parse()

# Access nested parameters
print(f"App: {Config.app_name}")
print(f"Database: {Config.database.host}:{Config.database.port}/{Config.database.name}")
print(f"Redis: {Config.redis.host}:{Config.redis.port}/{Config.redis.db}")
print(f"Debug: {Config.debug}")
```

### Command Line Usage

```bash
# Override top-level parameters
python app.py --Config.app_name "Production App" --Config.debug

# Override nested parameters
python app.py \
    --DatabaseConfig.host "prod-db.example.com" \
    --DatabaseConfig.port 5433 \
    --DatabaseConfig.name "production_db" \
    --RedisConfig.host "redis.example.com" \
    --RedisConfig.password "redis-secret"
```

## Prefix-Based Nested Configurations

Use prefixes to create cleaner command-line interfaces and better organization.

### Automatic Prefixes with PrefixProto

```python  
from params_proto.v2.proto import PrefixProto, ParamsProto, Proto

class ModelConfig(PrefixProto, prefix="model"):
    """Model architecture configuration"""
    architecture = Proto("Model architecture", default="resnet50")
    num_layers = Proto("Number of layers", default=18)
    hidden_size = Proto("Hidden layer size", default=512)
    dropout_rate = Proto("Dropout rate", default=0.1)
    batch_norm = Proto("Use batch normalization", default=True)

class TrainingConfig(PrefixProto, prefix="training"):
    """Training hyperparameters"""
    learning_rate = Proto("Learning rate", default=0.001)
    batch_size = Proto("Batch size", default=32)
    epochs = Proto("Number of epochs", default=100)
    optimizer = Proto("Optimizer type", default="adam")
    weight_decay = Proto("Weight decay", default=1e-4)
    scheduler = Proto("Learning rate scheduler", default="cosine")

class DataConfig(PrefixProto, prefix="data"):
    """Data loading and preprocessing"""
    dataset = Proto("Dataset name", default="cifar10")
    data_dir = Proto("Data directory", default="./data")
    num_workers = Proto("Data loader workers", default=4)
    augmentation = Proto("Data augmentation", default=True)
    validation_split = Proto("Validation split ratio", default=0.1)

class ExperimentConfig(ParamsProto):
    """Experiment configuration"""
    experiment_name = Proto("Experiment name", default="baseline")
    seed = Proto("Random seed", default=42)
    output_dir = Proto("Output directory", default="./outputs")
    save_checkpoints = Proto("Save model checkpoints", default=True)

# Parse all configurations
ModelConfig.parse()
TrainingConfig.parse()
DataConfig.parse()
ExperimentConfig.parse()

print(f"Model: {ModelConfig.architecture} with {ModelConfig.num_layers} layers")
print(f"Training: LR={TrainingConfig.learning_rate}, BS={TrainingConfig.batch_size}")
print(f"Data: {DataConfig.dataset} from {DataConfig.data_dir}")
print(f"Experiment: {ExperimentConfig.experiment_name}")
```

### Command Line with Prefixes

```bash
# Clean command line interface with prefixes
python train.py \
    --model.architecture "transformer" \
    --model.num_layers 12 \
    --model.hidden_size 768 \
    --training.learning_rate 0.0001 \
    --training.batch_size 64 \
    --training.optimizer "adamw" \
    --data.dataset "imagenet" \
    --data.augmentation \
    --ExperimentConfig.experiment_name "transformer_large"
```

## Dynamic Nested Configurations

Create configurations that adapt based on other parameter values.

### Conditional Nested Configuration

```python
from params_proto.v2.proto import ParamsProto, Proto

class BaseModelConfig(ParamsProto, cli=False):
    """Base model configuration"""
    dropout = Proto("Dropout rate", default=0.1)
    activation = Proto("Activation function", default="relu")

class CNNModelConfig(BaseModelConfig, cli=False):
    """CNN-specific configuration"""
    num_conv_layers = Proto("Number of conv layers", default=3)
    kernel_size = Proto("Convolution kernel size", default=3)
    num_filters = Proto("Number of filters", default=64)

class TransformerModelConfig(BaseModelConfig, cli=False):
    """Transformer-specific configuration"""
    num_attention_heads = Proto("Number of attention heads", default=8)
    attention_dropout = Proto("Attention dropout", default=0.1)
    feed_forward_dim = Proto("Feed-forward dimension", default=2048)

class AdaptiveConfig(ParamsProto, cli_parse=False):
    """Configuration that adapts based on model type"""
    
    # Model selection
    model_type = Proto("Model type", default="cnn")
    
    # Dataset parameters
    dataset = Proto("Dataset", default="cifar10")
    
    def __init__(self, _deps=None):
        """Initialize with conditional nested configs"""
        super().__init__(_deps)
        
        # Choose model config based on type
        if self.model_type == "cnn":
            self.model = CNNModelConfig(_deps)
        elif self.model_type == "transformer":
            self.model = TransformerModelConfig(_deps)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Set dataset-specific parameters
        if self.dataset == "cifar10":
            self.num_classes = 10
            self.input_size = 32
        elif self.dataset == "imagenet":
            self.num_classes = 1000
            self.input_size = 224
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

# Usage
AdaptiveConfig.parse()

# Create configuration instance
config = AdaptiveConfig()
print(f"Model type: {config.model_type}")
print(f"Dataset: {config.dataset}")
print(f"Classes: {config.num_classes}")

if config.model_type == "cnn":
    print(f"Conv layers: {config.model.num_conv_layers}")
    print(f"Filters: {config.model.num_filters}")
elif config.model_type == "transformer":
    print(f"Attention heads: {config.model.num_attention_heads}")
    print(f"FF dim: {config.model.feed_forward_dim}")

# Command line usage:
# python train.py --AdaptiveConfig.model_type transformer --AdaptiveConfig.dataset imagenet
```

### Multi-Stage Nested Configurations

```python
class InfrastructureConfig(ParamsProto, cli_parse=False):
    """Infrastructure configuration that affects others"""
    deployment_type = Proto("Deployment type", default="local")
    environment = Proto("Environment", default="development")
    cloud_provider = Proto("Cloud provider", default="aws")

class ComputeConfig(ParamsProto, cli=False):
    """Compute resource configuration"""
    
    def __init__(self, infra_config):
        super().__init__()
        
        if infra_config.deployment_type == "local":
            self.num_gpus = 1
            self.num_workers = 2
            self.memory_gb = 8
        elif infra_config.deployment_type == "cluster":
            self.num_gpus = 4
            self.num_workers = 16
            self.memory_gb = 64
        elif infra_config.deployment_type == "cloud":
            if infra_config.cloud_provider == "aws":
                self.instance_type = "p3.2xlarge"
                self.num_gpus = 1
            elif infra_config.cloud_provider == "gcp":
                self.instance_type = "n1-highmem-8"
                self.num_gpus = 2

class StorageConfig(ParamsProto, cli=False):
    """Storage configuration"""
    
    def __init__(self, infra_config):
        super().__init__()
        
        if infra_config.deployment_type == "local":
            self.data_path = "./data"
            self.output_path = "./outputs"
            self.cache_path = "/tmp/cache"
        elif infra_config.deployment_type == "cluster":
            self.data_path = "/shared/data"
            self.output_path = "/shared/outputs"
            self.cache_path = "/local/cache"
        elif infra_config.deployment_type == "cloud":
            if infra_config.cloud_provider == "aws":
                self.data_path = "s3://mybucket/data"
                self.output_path = "s3://mybucket/outputs"
                self.cache_path = "/tmp/cache"

class ApplicationConfig(ParamsProto):
    """Main application configuration"""
    
    # Application parameters
    app_name = Proto("Application name", default="MLApp")
    
    def __init__(self, _deps=None):
        # Initialize infrastructure config first
        self.infra = InfrastructureConfig(_deps)
        
        # Initialize dependent configs
        self.compute = ComputeConfig(self.infra)
        self.storage = StorageConfig(self.infra)
        
        # Set application-specific parameters based on infrastructure
        if self.infra.environment == "production":
            self.log_level = "INFO"
            self.debug = False
        else:
            self.log_level = "DEBUG"
            self.debug = True

# Usage
ApplicationConfig.parse()
app_config = ApplicationConfig()

print(f"Deployment: {app_config.infra.deployment_type}")
print(f"Environment: {app_config.infra.environment}")
print(f"GPUs: {app_config.compute.num_gpus}")
print(f"Workers: {app_config.compute.num_workers}")
print(f"Data path: {app_config.storage.data_path}")
print(f"Debug: {app_config.debug}")

# Override infrastructure settings:
# python app.py \
#   --InfrastructureConfig.deployment_type cluster \
#   --InfrastructureConfig.environment production
```

## Complex Nested Hierarchies

### Deep Nesting Example

```python
class DatabasePoolConfig(ParamsProto, cli=False):
    """Database connection pool settings"""
    min_connections = Proto("Minimum connections", default=1)
    max_connections = Proto("Maximum connections", default=10)
    connection_timeout = Proto("Connection timeout (seconds)", default=30)

class DatabaseConfig(ParamsProto, cli=False):
    """Database configuration with nested pool settings"""
    host = Proto("Database host", default="localhost")
    port = Proto("Database port", default=5432)
    name = Proto("Database name", default="myapp")
    user = Proto("Database user", default="postgres")
    password = Proto("Database password", default="password")
    
    # Nested pool configuration
    pool = DatabasePoolConfig

class CacheRedisConfig(ParamsProto, cli=False):
    """Redis configuration for caching"""
    host = Proto("Redis host", default="localhost")
    port = Proto("Redis port", default=6379)
    db = Proto("Redis database", default=0)

class SessionRedisConfig(ParamsProto, cli=False):
    """Redis configuration for sessions"""
    host = Proto("Redis host", default="localhost")
    port = Proto("Redis port", default=6379)
    db = Proto("Redis database", default=1)

class CacheConfig(ParamsProto, cli=False):
    """Caching configuration"""
    enabled = Proto("Enable caching", default=True)
    ttl_seconds = Proto("Default TTL in seconds", default=3600)
    
    # Nested Redis config
    redis = CacheRedisConfig

class SessionConfig(ParamsProto, cli=False):
    """Session management configuration"""
    secret_key = Proto("Session secret key", default="change-me")
    expire_minutes = Proto("Session expiry in minutes", default=60)
    
    # Nested Redis config
    redis = SessionRedisConfig

class AppConfig(ParamsProto):
    """Application configuration with deep nesting"""
    
    # Top-level app settings
    name = Proto("Application name", default="MyApp")
    version = Proto("Application version", default="1.0.0")
    
    # Nested configurations
    database = DatabaseConfig
    cache = CacheConfig
    session = SessionConfig

# Parse configuration
AppConfig.parse()

# Access deeply nested parameters
print(f"App: {AppConfig.name} v{AppConfig.version}")
print(f"Database: {AppConfig.database.host}:{AppConfig.database.port}")
print(f"DB Pool: {AppConfig.database.pool.min_connections}-{AppConfig.database.pool.max_connections}")
print(f"Cache Redis: {AppConfig.cache.redis.host}:{AppConfig.cache.redis.port}/{AppConfig.cache.redis.db}")
print(f"Session Redis: {AppConfig.session.redis.host}:{AppConfig.session.redis.port}/{AppConfig.session.redis.db}")
```

### Command Line for Deep Nesting

```bash
# Override deeply nested parameters
python app.py \
    --AppConfig.name "Production App" \
    --DatabaseConfig.host "prod-db.company.com" \
    --DatabasePoolConfig.max_connections 50 \
    --CacheRedisConfig.host "cache-redis.company.com" \
    --CacheRedisConfig.db 2 \
    --SessionRedisConfig.host "session-redis.company.com" \
    --CacheConfig.ttl_seconds 7200
```

## Configuration Composition

### Mixing Inheritance and Composition

```python
class BaseServiceConfig(ParamsProto, cli=False):
    """Base configuration for all services"""
    service_name = Proto("Service name", default="unknown")
    port = Proto("Service port", default=8000)
    host = Proto("Service host", default="0.0.0.0")
    workers = Proto("Number of workers", default=1)

class APIServiceConfig(BaseServiceConfig):
    """API service specific configuration"""
    api_version = Proto("API version", default="v1")
    rate_limit_per_minute = Proto("Rate limit per minute", default=60)
    enable_cors = Proto("Enable CORS", default=True)

class WebServiceConfig(BaseServiceConfig):
    """Web service specific configuration"""
    static_files_dir = Proto("Static files directory", default="./static")
    template_dir = Proto("Templates directory", default="./templates")
    session_timeout = Proto("Session timeout minutes", default=30)

class MicroservicesConfig(ParamsProto):
    """Configuration for microservices architecture"""
    
    # Service configurations
    api_service = APIServiceConfig
    web_service = WebServiceConfig
    
    # Shared configurations
    database_url = Proto("Shared database URL", default="postgresql://localhost/shared")
    redis_url = Proto("Shared Redis URL", default="redis://localhost:6379")
    
    # Load balancer configuration
    lb_enabled = Proto("Enable load balancer", default=False)
    lb_algorithm = Proto("Load balancing algorithm", default="round_robin")

# Initialize with custom values
MicroservicesConfig.parse()

# Override service-specific settings
MicroservicesConfig.api_service._update(
    service_name="user-api",
    port=8001,
    workers=4
)

MicroservicesConfig.web_service._update(
    service_name="web-frontend", 
    port=8002,
    workers=2
)

print("=== API Service ===")
print(f"Name: {MicroservicesConfig.api_service.service_name}")
print(f"Port: {MicroservicesConfig.api_service.port}")
print(f"API Version: {MicroservicesConfig.api_service.api_version}")
print(f"Workers: {MicroservicesConfig.api_service.workers}")

print("\n=== Web Service ===")
print(f"Name: {MicroservicesConfig.web_service.service_name}")
print(f"Port: {MicroservicesConfig.web_service.port}")
print(f"Static Dir: {MicroservicesConfig.web_service.static_files_dir}")
print(f"Workers: {MicroservicesConfig.web_service.workers}")

print("\n=== Shared ===")
print(f"Database: {MicroservicesConfig.database_url}")
print(f"Redis: {MicroservicesConfig.redis_url}")
```

## Best Practices for Nested Configurations

### Organization Guidelines

```python
# Good: Logical grouping by functionality
class DatabaseConfig(ParamsProto, cli=False):
    host = Proto("Database host", default="localhost")
    port = Proto("Database port", default=5432)
    name = Proto("Database name", default="myapp")

class AuthConfig(ParamsProto, cli=False):
    jwt_secret = Proto("JWT secret", default="secret")
    token_expiry = Proto("Token expiry hours", default=24)

class Config(ParamsProto):
    database = DatabaseConfig
    auth = AuthConfig

# Avoid: Too deep nesting (more than 3 levels)
# Avoid: Single parameter per config class
# Avoid: Mixing concerns in one config class
```

### Documentation and Help

```python
class WellDocumentedConfig(ParamsProto):
    """
    Main application configuration.
    
    This configuration supports both command-line arguments and environment variables.
    Use --help to see all available options.
    """
    
    class Database(ParamsProto, cli=False):
        """
        Database connection settings.
        
        Supports PostgreSQL, MySQL, and SQLite databases.
        """
        url = Proto("Database connection URL", 
                   default="sqlite:///app.db",
                   help="Full database URL including credentials")
        
        pool_size = Proto("Connection pool size",
                         default=5,
                         help="Maximum number of database connections")
    
    class Logging(ParamsProto, cli=False):
        """
        Application logging configuration.
        
        Controls log levels, formats, and output destinations.
        """
        level = Proto("Log level", 
                     default="INFO",
                     help="Minimum log level (DEBUG, INFO, WARNING, ERROR)")
        
        format = Proto("Log format string",
                      default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                      help="Python logging format string")
    
    # Nested configurations with clear documentation
    database = Database
    logging = Logging
    
    # Top-level settings
    app_name = Proto("Application name", 
                    default="MyApp",
                    help="Human-readable application name")

# Usage
if __name__ == "__main__":
    WellDocumentedConfig.parse()
    
    # Access configuration
    print(f"App: {WellDocumentedConfig.app_name}")
    print(f"Database: {WellDocumentedConfig.database.url}")
    print(f"Log level: {WellDocumentedConfig.logging.level}")
```

This comprehensive guide shows how to effectively use nested configurations in params-proto for organizing complex application settings. Nested configurations provide better structure, clearer command-line interfaces, and more maintainable code for large applications.