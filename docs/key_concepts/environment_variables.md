# Environment Variables

Commandline arguments are great and that, but remember the last time
when you wanted to use environment variables? Here we provide the `EnvVar` 
class to help you define default values via environment flags.

params-proto' `EnvVar` automatically handles type conversion and template expansion. 
Enjoy!

## Quick Start

```python
from params_proto import proto, EnvVar

@proto.cli
def train(
    # Read from environment variable with fallback
    batch_size: int = EnvVar @ "BATCH_SIZE" | 128,

    # Read from environment variable (no fallback)
    api_key: str = EnvVar @ "API_KEY",

    # Template expansion with multiple variables
    data_dir: str = EnvVar @ "$HOME/data/$PROJECT",
):
    """Train model with environment configuration."""
    print(batch_size, api_key, data_dir)
```

**Usage:**
```bash
# Set environment variables
export BATCH_SIZE=256
export API_KEY=secret-key-123
export HOME=/home/alice
export PROJECT=ml-project

# Run with env vars
python train.py

# CLI args can still override env vars
python train.py --batch-size 512
```

## Usage Patterns

### 1. Matmul Operator (`@`)

The cleanest syntax uses the `@` operator:

```python
@proto.cli
def config(
    # Environment variable only (no fallback)
    port: int = EnvVar @ "PORT",

    # Environment variable with fallback using | operator
    host: str = EnvVar @ "HOST" | "localhost",

    # Template expansion
    log_file: str = EnvVar @ "$HOME/logs/app.log",
):
    """Configuration from environment."""
    pass
```

**CLI usage:**
```bash
# Without env vars - uses fallbacks
python config.py
# port=None, host="localhost", log_file expands $HOME

# With env vars set
export PORT=8080
export HOST=api.example.com
export HOME=/home/user
python config.py
# port=8080, host="api.example.com", log_file="/home/user/logs/app.log"
```

### 2. Function Call Syntax

Use function call syntax for explicit keyword arguments:

```python
@proto.cli
def connect(
    db_url: str = EnvVar("DATABASE_URL", default="sqlite:///local.db"),
    timeout: int = EnvVar("DB_TIMEOUT", default=30),
    pool_size: int = EnvVar("DB_POOL_SIZE", default=10),
):
    """Connect to database with environment configuration."""
    print(f"Connecting to: {db_url}")
```

**CLI usage:**
```bash
# Uses all defaults
python connect.py
# db_url="sqlite:///local.db", timeout=30, pool_size=10

# With environment variables
export DATABASE_URL=postgres://prod-db:5432/myapp
export DB_TIMEOUT=60
python connect.py
# db_url="postgres://prod-db:5432/myapp", timeout=60, pool_size=10
```

### 3. Pipe Operator for Defaults

The pipe operator (`|`) provides a clean syntax for fallback values:

```python
@proto.cli
def train(
    # Pipe operator chains env var name with fallback
    lr: float = EnvVar @ "LEARNING_RATE" | 0.001,
    epochs: int = EnvVar @ "EPOCHS" | 100,
    seed: int = EnvVar @ "RANDOM_SEED" | 42,
):
    """Train with environment or defaults."""
    pass
```

**Behavior:**
- If environment variable exists: uses its value (with type conversion)
- If environment variable missing: uses the fallback value

## Template Expansion

EnvVar supports template strings with environment variable substitution.

### Basic Template Syntax

Two syntaxes are supported:

```python
@proto.cli
def setup(
    # Dollar prefix: $VAR_NAME
    home_dir: str = EnvVar @ "$HOME",

    # Braces syntax: ${VAR_NAME}
    config_file: str = EnvVar("${HOME}/.config/app.conf", default="~/.config/app.conf"),

    # Both syntaxes work together
    log_path: str = EnvVar @ "$HOME/logs/${APP_NAME}.log",
):
    """Setup with path templates."""
    pass
```

**Example:**
```bash
export HOME=/home/alice
export APP_NAME=trainer

python setup.py
# home_dir="/home/alice"
# config_file="/home/alice/.config/app.conf"
# log_path="/home/alice/logs/trainer.log"
```

### Multiple Variables in Templates

Combine multiple environment variables in a single template:

```python
@proto.cli
def configure(
    # Multiple variables in one path
    workspace: str = EnvVar("/home/${USER}/projects/${PROJECT}", default="/tmp"),

    # Complex checkpoint directory
    checkpoint_dir: str = EnvVar("$BASE_DIR/${PROJECT}/${EXPERIMENT}/checkpoints", default="/tmp/checkpoints"),

    # Log file with multiple vars
    log_file: str = EnvVar("${USER}_${PROJECT}_$EXPERIMENT.log", default="default.log"),
):
    """Configure paths with multiple environment variables."""
    pass
```

**Example:**
```bash
export USER=alice
export PROJECT=ml-research
export EXPERIMENT=exp001
export BASE_DIR=/data

python configure.py
# workspace="/home/alice/projects/ml-research"
# checkpoint_dir="/data/ml-research/exp001/checkpoints"
# log_file="alice_ml-research_exp001.log"
```

### Missing Variables in Templates

When a variable in a template is not set, it's replaced with an empty string:

```python
@proto.cli
def get_path(
    # SUBDIR might not be set
    data_path: str = EnvVar("$BASE_DIR/$SUBDIR/output", default="/tmp/output"),
):
    """Get path with potentially missing vars."""
    pass
```

**Example:**
```bash
export BASE_DIR=/data
# SUBDIR is NOT set

python get_path.py
# data_path="/data//output"  (SUBDIR becomes empty string)

export SUBDIR=training
python get_path.py
# data_path="/data/training/output"
```

## Type Conversion

EnvVar automatically converts string values from environment variables to the annotated type:

```python
@proto.cli
def config(
    # String to int
    port: int = EnvVar @ "PORT" | 8080,

    # String to float
    threshold: float = EnvVar @ "THRESHOLD" | 0.75,

    # String to bool
    debug: bool = EnvVar @ "DEBUG" | False,

    # Remains string
    api_key: str = EnvVar @ "API_KEY" | "dev-key",
):
    """Configuration with type conversion."""
    pass
```

**Example:**
```bash
export PORT=3000
export THRESHOLD=0.95
export DEBUG=true
export API_KEY=prod-key-123

python config.py
# port=3000 (int)
# threshold=0.95 (float)
# debug=True (bool)
# api_key="prod-key-123" (str)
```

**Boolean conversion rules:**
- `"true"`, `"True"`, `"1"`, `"yes"` → `True`
- `"false"`, `"False"`, `"0"`, `"no"`, `""` → `False`

## Override Precedence

Environment variables are resolved during decoration and follow the params-proto override hierarchy:

1. **CLI arguments** (highest priority)
2. **`proto.bind()` context**
3. **Direct assignment** (`train.lr = 0.01`)
4. **Environment variables** ← EnvVar resolution
5. **Default values** (lowest priority)

```python
@proto.cli
def train(
    lr: float = EnvVar @ "LEARNING_RATE" | 0.001,
):
    """Train with learning rate."""
    print(f"LR: {lr}")

# Environment variable
# export LEARNING_RATE=0.01

# CLI arg overrides env var
train()  # Uses 0.01 from LEARNING_RATE env var

# But CLI args have higher priority
# python train.py --lr 0.1  # Uses 0.1 from CLI
```

## Common Patterns

### Pattern 1: Configuration from Environment

Production deployments often use environment variables for secrets and configuration:

```python
@proto.cli
def api_server(
    # Database connection
    db_url: str = EnvVar @ "DATABASE_URL" | "sqlite:///dev.db",
    db_pool_size: int = EnvVar @ "DB_POOL_SIZE" | 10,

    # API keys and secrets
    api_key: str = EnvVar @ "API_KEY",  # Required, no default
    secret_key: str = EnvVar @ "SECRET_KEY",

    # Server settings
    port: int = EnvVar @ "PORT" | 8000,
    host: str = EnvVar @ "HOST" | "0.0.0.0",
    workers: int = EnvVar @ "WORKERS" | 4,

    # Feature flags
    debug: bool = EnvVar @ "DEBUG" | False,
    enable_cors: bool = EnvVar @ "ENABLE_CORS" | True,
):
    """API server with environment configuration."""
    print(f"Starting server on {host}:{port}")
    print(f"Workers: {workers}, Debug: {debug}")
```

### Pattern 2: Path Configuration

Use templates for flexible path configuration:

```python
@proto.cli
def ml_pipeline(
    # Base directories from environment
    data_dir: str = EnvVar @ "$DATA_DIR/datasets" | "./data",
    model_dir: str = EnvVar @ "$MODEL_DIR/checkpoints" | "./models",
    log_dir: str = EnvVar @ "$LOG_DIR/experiments" | "./logs",

    # Experiment-specific paths with templates
    experiment_dir: str = EnvVar @ "$LOG_DIR/${EXPERIMENT_NAME}",
    checkpoint_path: str = EnvVar @ "$MODEL_DIR/${EXPERIMENT_NAME}/best.pt",
):
    """ML pipeline with environment-based paths."""
    print(f"Data: {data_dir}")
    print(f"Experiment: {experiment_dir}")
```

### Pattern 3: Development vs Production

Use environment variables to switch between development and production configurations:

```python
@proto.prefix
class Config:
    """Application configuration."""
    # Environment selector
    env: str = EnvVar @ "APP_ENV" | "development"

    # Database (different per environment)
    db_url: str = EnvVar @ "DATABASE_URL" | "sqlite:///dev.db"

    # Redis cache
    redis_url: str = EnvVar @ "REDIS_URL" | "redis://localhost:6379"

    # Logging
    log_level: str = EnvVar @ "LOG_LEVEL" | "INFO"

    # Feature flags
    enable_metrics: bool = EnvVar @ "ENABLE_METRICS" | False
    enable_tracing: bool = EnvVar @ "ENABLE_TRACING" | False

@proto.cli
def main():
    """Run application."""
    if Config.env == "production":
        print("Running in PRODUCTION mode")
        assert Config.db_url != "sqlite:///dev.db", "Must set DATABASE_URL for production"
    else:
        print("Running in DEVELOPMENT mode")
```

**Development:**
```bash
python main.py
# Uses all defaults
```

**Production:**
```bash
export APP_ENV=production
export DATABASE_URL=postgres://prod-db:5432/myapp
export REDIS_URL=redis://prod-cache:6379
export LOG_LEVEL=WARNING
export ENABLE_METRICS=true
export ENABLE_TRACING=true

python main.py
# Uses production configuration
```

### Pattern 4: Container Orchestration

Kubernetes and Docker Compose often inject configuration via environment variables:

```python
@proto.prefix
class K8sConfig:
    """Kubernetes-style configuration."""
    # Pod information
    pod_name: str = EnvVar @ "POD_NAME" | "local"
    pod_namespace: str = EnvVar @ "POD_NAMESPACE" | "default"
    pod_ip: str = EnvVar @ "POD_IP" | "127.0.0.1"

    # Service discovery
    service_host: str = EnvVar @ "${SERVICE_NAME}_SERVICE_HOST" | "localhost"
    service_port: int = EnvVar @ "${SERVICE_NAME}_SERVICE_PORT" | 8080

    # Secrets (mounted as env vars)
    db_password: str = EnvVar @ "DB_PASSWORD"
    api_token: str = EnvVar @ "API_TOKEN"

@proto.cli
def worker():
    """Kubernetes worker pod."""
    print(f"Pod: {K8sConfig.pod_name} in {K8sConfig.pod_namespace}")
    print(f"Connecting to service at {K8sConfig.service_host}:{K8sConfig.service_port}")
```

### Pattern 5: Inheritance with EnvVar

EnvVar fields work correctly with class inheritance. Inherited fields are resolved and type-converted:

```python
class BaseConfig:
    """Common configuration shared across services."""
    host: str = EnvVar @ "HOST" | "localhost"
    port: int = EnvVar @ "PORT" | 8080
    debug: bool = EnvVar @ "DEBUG" | False
    log_level: str = EnvVar @ "LOG_LEVEL" | "INFO"

@proto.prefix
class APIConfig(BaseConfig):
    """API service configuration."""
    timeout: int = EnvVar @ "API_TIMEOUT" | 30
    max_retries: int = EnvVar @ "API_MAX_RETRIES" | 3
    api_key: str = EnvVar @ "API_KEY"

@proto.prefix
class WorkerConfig(BaseConfig):
    """Background worker configuration."""
    concurrency: int = EnvVar @ "WORKER_CONCURRENCY" | 4
    queue_name: str = EnvVar @ "WORKER_QUEUE" | "default"
```

Both inherited and child EnvVar fields:
- Resolve from environment variables at decoration time
- Convert to the annotated type (`str`, `int`, `bool`, `float`)
- Use fallback defaults when the env var is not set

**Usage:**
```bash
# Shared config applies to both services
export HOST=10.0.0.1
export PORT=3000
export DEBUG=true

# Service-specific config
export API_TIMEOUT=60
export API_KEY=secret-key
export WORKER_CONCURRENCY=8

python api_server.py   # Uses APIConfig with inherited HOST, PORT, DEBUG
python worker.py       # Uses WorkerConfig with same inherited fields
```

## Security Considerations

### 1. Never Log Secrets

Be careful not to log environment variables that contain secrets:

```python
# ✗ BAD: Logs the secret
@proto.cli
def connect(api_key: str = EnvVar @ "API_KEY"):
    print(f"Using API key: {api_key}")  # ✗ Exposes secret in logs

# ✓ GOOD: Masks the secret
@proto.cli
def connect(api_key: str = EnvVar @ "API_KEY"):
    masked = api_key[:4] + "..." if api_key else "None"
    print(f"Using API key: {masked}")  # ✓ Safe logging
```

### 2. Require Secrets in Production

Use assertions to ensure required secrets are set:

```python
@proto.prefix
class Config:
    env: str = EnvVar @ "APP_ENV" | "development"
    secret_key: str = EnvVar @ "SECRET_KEY" | "dev-secret-key"

@proto.cli
def main():
    """Run application."""
    if Config.env == "production":
        assert Config.secret_key != "dev-secret-key", (
            "SECRET_KEY environment variable must be set in production"
        )
```

### 3. Validate Environment Variables

Add validation for critical configuration:

```python
@proto.cli
def server(
    port: int = EnvVar @ "PORT" | 8000,
    workers: int = EnvVar @ "WORKERS" | 4,
):
    """Start server with validation."""
    assert 1024 <= port <= 65535, f"Port must be in range 1024-65535, got {port}"
    assert workers > 0, f"Workers must be positive, got {workers}"

    print(f"Starting server on port {port} with {workers} workers")
```

## Testing with Environment Variables

When testing code that uses `EnvVar`, manage environment variables in test fixtures:

```python
import os
import pytest
from params_proto import proto, EnvVar

def test_envvar_configuration():
    """Test configuration from environment variables."""
    # Set up test environment
    os.environ["TEST_PORT"] = "9000"
    os.environ["TEST_HOST"] = "testhost"

    try:
        @proto
        def config(
            port: int = EnvVar @ "TEST_PORT" | 8000,
            host: str = EnvVar @ "TEST_HOST" | "localhost",
        ):
            return port, host

        # Test with env vars
        result = config()
        assert result == (9000, "testhost")

    finally:
        # Clean up
        del os.environ["TEST_PORT"]
        del os.environ["TEST_HOST"]
```

**Better approach with pytest fixtures:**

```python
@pytest.fixture
def test_env():
    """Set up test environment variables."""
    original = os.environ.copy()

    # Set test vars
    os.environ["TEST_PORT"] = "9000"
    os.environ["TEST_HOST"] = "testhost"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original)

def test_config(test_env):
    """Test with environment fixture."""
    @proto
    def config(
        port: int = EnvVar @ "TEST_PORT",
        host: str = EnvVar @ "TEST_HOST",
    ):
        return port, host

    assert config() == (9000, "testhost")
```

## Related

- [Core Concepts](core-concepts) - Decorators and basic usage
- [CLI Fundamentals](cli-fundamentals) - Basic CLI features
- [Configuration Patterns](configuration-patterns) - Function vs class configurations
- [Type System](type-system) - Type conversion rules
- [Parameter Overrides](parameter-overrides) - Override precedence and context managers
