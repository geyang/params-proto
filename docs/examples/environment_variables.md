# Environment Variables

Learn how to use environment variables with params-proto for flexible configuration management across different deployment environments.

## Basic Environment Variable Usage

Environment variables provide a way to configure your application without changing code, making it perfect for different deployment environments.

### Simple Environment Variables

```python
from params_proto.v2.proto import ParamsProto, Proto
import os

class Config(ParamsProto):
    # Basic environment variable with fallback
    app_name = Proto("Application name", env="APP_NAME", default="MyApp")
    
    # Numeric environment variable with type conversion
    port = Proto("Server port", env="PORT", dtype=int, default=8000)
    
    # Boolean environment variable
    debug = Proto("Debug mode", env="DEBUG", dtype=bool, default=False)
    
    # Required environment variable (fails if not set)
    secret_key = Proto("Secret key", env="SECRET_KEY", strict_parsing=True)

# Set some environment variables
os.environ['APP_NAME'] = 'Production App'
os.environ['PORT'] = '9000'
os.environ['DEBUG'] = 'true'
os.environ['SECRET_KEY'] = 'super-secret-key'

# Parse configuration
Config.parse()

print(f"App: {Config.app_name}")      # Production App
print(f"Port: {Config.port}")         # 9000 (converted to int)
print(f"Debug: {Config.debug}")       # True (converted to bool)
print(f"Secret: {Config.secret_key}") # super-secret-key
```

### Variable Expansion Syntax

Use `${VAR_NAME}` syntax for more flexible environment variable handling:

```python
class PathConfig(ParamsProto):
    # Environment variable with fallback value
    data_dir = Proto("Data directory", default="${DATA_DIR:/tmp/data}")
    
    # Combine multiple environment variables
    log_file = Proto("Log file", default="${LOG_DIR:/var/log}/${APP_NAME:myapp}.log")
    
    # User home directory expansion
    config_file = Proto("Config file", default="${HOME}/.myapp/config.yml")
    
    # Complex fallback chain
    cache_dir = Proto("Cache directory", default="${CACHE_DIR:${TMPDIR:/tmp}}/cache")

# The variables are expanded automatically
PathConfig.parse()

print(f"Data dir: {PathConfig.data_dir}")
print(f"Log file: {PathConfig.log_file}")
print(f"Config: {PathConfig.config_file}")
print(f"Cache: {PathConfig.cache_dir}")
```

## Deployment Environment Patterns

### Development vs Production

```python
class AppConfig(ParamsProto):
    # Environment detection
    environment = Proto("Environment", env="ENVIRONMENT", default="development")
    
    # Database configuration
    db_host = Proto("Database host", env="DB_HOST", default="localhost")
    db_port = Proto("Database port", env="DB_PORT", dtype=int, default=5432)
    db_name = Proto("Database name", env="DB_NAME", default="myapp_dev")
    db_user = Proto("Database user", env="DB_USER", default="postgres")
    db_password = Proto("Database password", env="DB_PASSWORD", strict_parsing=True)
    
    # Redis configuration
    redis_url = Proto("Redis URL", default="${REDIS_URL:redis://localhost:6379}")
    
    # Logging
    log_level = Proto("Log level", env="LOG_LEVEL", default="DEBUG")
    
    # Feature flags
    enable_analytics = Proto("Enable analytics", env="ENABLE_ANALYTICS", dtype=bool, default=False)
    enable_caching = Proto("Enable caching", env="ENABLE_CACHING", dtype=bool, default=True)

# Development environment (uses defaults)
# No environment variables needed

# Staging environment
# export ENVIRONMENT=staging
# export DB_HOST=staging-db.company.com
# export DB_NAME=myapp_staging
# export DB_PASSWORD=staging-password
# export LOG_LEVEL=INFO
# export ENABLE_ANALYTICS=true

# Production environment  
# export ENVIRONMENT=production
# export DB_HOST=prod-db.company.com
# export DB_NAME=myapp_prod
# export DB_PASSWORD=secure-prod-password
# export REDIS_URL=redis://prod-redis.company.com:6379
# export LOG_LEVEL=WARNING
# export ENABLE_ANALYTICS=true
```

### Docker Integration

Create a configuration that works well with Docker containers:

```python
class DockerConfig(ParamsProto):
    # Service configuration
    service_name = Proto("Service name", env="SERVICE_NAME", default="myapp")
    service_port = Proto("Service port", env="PORT", dtype=int, default=8000)
    
    # External services (typically provided by Docker Compose)
    postgres_host = Proto("PostgreSQL host", env="POSTGRES_HOST", default="postgres")
    postgres_port = Proto("PostgreSQL port", env="POSTGRES_PORT", dtype=int, default=5432)
    postgres_db = Proto("PostgreSQL database", env="POSTGRES_DB", default="myapp")
    postgres_user = Proto("PostgreSQL user", env="POSTGRES_USER", default="postgres")
    postgres_password = Proto("PostgreSQL password", env="POSTGRES_PASSWORD", strict_parsing=True)
    
    redis_host = Proto("Redis host", env="REDIS_HOST", default="redis")  
    redis_port = Proto("Redis port", env="REDIS_PORT", dtype=int, default=6379)
    
    # Observability
    jaeger_agent_host = Proto("Jaeger agent", env="JAEGER_AGENT_HOST", default="jaeger")
    
    # Build information (set by CI/CD)
    git_commit = Proto("Git commit", env="GIT_COMMIT", default="unknown")
    build_timestamp = Proto("Build timestamp", env="BUILD_TIMESTAMP", default="unknown")

def get_database_url():
    """Construct database URL from individual components"""
    DockerConfig.parse()
    return (f"postgresql://{DockerConfig.postgres_user}:{DockerConfig.postgres_password}@"
            f"{DockerConfig.postgres_host}:{DockerConfig.postgres_port}/{DockerConfig.postgres_db}")

def get_redis_url():
    """Construct Redis URL"""
    return f"redis://{DockerConfig.redis_host}:{DockerConfig.redis_port}"
```

**Docker Compose example:**

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    environment:
      - SERVICE_NAME=myapp
      - PORT=8000
      - POSTGRES_HOST=postgres
      - POSTGRES_PASSWORD=dev_password
      - REDIS_HOST=redis
      - GIT_COMMIT=${GIT_COMMIT:-dev}
    depends_on:
      - postgres
      - redis
  
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=dev_password
    
  redis:
    image: redis:6-alpine
```

## Environment-Specific Configurations

### Multi-Environment Setup

```python
class MultiEnvConfig(ParamsProto):
    # Core environment setting
    env = Proto("Environment", env="ENV", default="development")
    
    # Base URLs that change per environment
    api_base_url = Proto("API base URL", 
                        default="${API_BASE_URL:http://localhost:3000}")
    
    web_base_url = Proto("Web base URL",
                        default="${WEB_BASE_URL:http://localhost:8080}")
    
    # Database connections
    database_url = Proto("Database URL",
                        default="${DATABASE_URL:sqlite:///dev.db}")
    
    # External service configurations
    stripe_api_key = Proto("Stripe API key", env="STRIPE_API_KEY")
    sendgrid_api_key = Proto("SendGrid API key", env="SENDGRID_API_KEY")
    
    # Feature flags per environment
    enable_debug_toolbar = Proto("Debug toolbar", 
                               env="ENABLE_DEBUG_TOOLBAR", 
                               dtype=bool, default=True)
    
    enable_profiling = Proto("Enable profiling",
                           env="ENABLE_PROFILING",
                           dtype=bool, default=False)
    
    # Performance settings
    worker_count = Proto("Worker processes", 
                        env="WORKER_COUNT", 
                        dtype=int, default=1)
    
    max_connections = Proto("Max DB connections",
                          env="MAX_DB_CONNECTIONS",
                          dtype=int, default=10)

# Environment-specific .env files:

# .env.development
# ENV=development
# API_BASE_URL=http://localhost:3000
# WEB_BASE_URL=http://localhost:8080  
# DATABASE_URL=sqlite:///dev.db
# ENABLE_DEBUG_TOOLBAR=true
# ENABLE_PROFILING=false
# WORKER_COUNT=1

# .env.staging
# ENV=staging
# API_BASE_URL=https://api-staging.company.com
# WEB_BASE_URL=https://staging.company.com
# DATABASE_URL=postgresql://user:pass@staging-db.company.com/myapp
# STRIPE_API_KEY=sk_test_...
# SENDGRID_API_KEY=SG.staging...
# ENABLE_DEBUG_TOOLBAR=false
# ENABLE_PROFILING=true
# WORKER_COUNT=2
# MAX_DB_CONNECTIONS=20

# .env.production
# ENV=production
# API_BASE_URL=https://api.company.com
# WEB_BASE_URL=https://company.com
# DATABASE_URL=postgresql://user:pass@prod-db.company.com/myapp
# STRIPE_API_KEY=sk_live_...
# SENDGRID_API_KEY=SG.production...
# ENABLE_DEBUG_TOOLBAR=false
# ENABLE_PROFILING=false
# WORKER_COUNT=4
# MAX_DB_CONNECTIONS=50
```

### Environment Loading Utilities

```python
import os
from pathlib import Path

def load_env_file(env_file=None):
    """Load environment variables from .env file"""
    if env_file is None:
        # Try to detect environment and load appropriate file
        env = os.environ.get('ENV', 'development')
        env_file = f'.env.{env}'
    
    env_path = Path(env_file)
    if not env_path.exists():
        print(f"Warning: {env_file} not found")
        return
    
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value
    
    print(f"Loaded environment from {env_file}")

def setup_environment():
    """Setup environment configuration"""
    # Load environment file
    load_env_file()
    
    # Parse configuration
    MultiEnvConfig.parse()
    
    # Print loaded configuration
    print(f"Environment: {MultiEnvConfig.env}")
    print(f"API URL: {MultiEnvConfig.api_base_url}")
    print(f"Database: {MultiEnvConfig.database_url}")
    print(f"Workers: {MultiEnvConfig.worker_count}")

# Usage
if __name__ == "__main__":
    setup_environment()
```

## Secrets Management

### Environment Variables for Secrets

```python
class SecretsConfig(ParamsProto):
    # Database credentials
    db_password = Proto("Database password", 
                       env="DB_PASSWORD", 
                       strict_parsing=True)
    
    # API keys
    jwt_secret = Proto("JWT secret key",
                      env="JWT_SECRET_KEY",
                      strict_parsing=True)
    
    oauth_client_secret = Proto("OAuth client secret",
                               env="OAUTH_CLIENT_SECRET",
                               strict_parsing=True)
    
    # External service keys
    aws_access_key = Proto("AWS access key", env="AWS_ACCESS_KEY_ID")
    aws_secret_key = Proto("AWS secret key", env="AWS_SECRET_ACCESS_KEY")
    
    # Encryption keys
    encryption_key = Proto("Encryption key", env="ENCRYPTION_KEY", strict_parsing=True)

def validate_secrets():
    """Validate that all required secrets are present"""
    try:
        SecretsConfig.parse()
        print("✓ All secrets loaded successfully")
        
        # Additional validation
        assert len(SecretsConfig.jwt_secret) >= 32, "JWT secret must be at least 32 characters"
        assert len(SecretsConfig.encryption_key) == 32, "Encryption key must be exactly 32 bytes"
        
        print("✓ Secret validation passed")
        
    except Exception as e:
        print(f"✗ Secret validation failed: {e}")
        raise

# Production secret loading
def load_secrets_from_vault():
    """Example: Load secrets from HashiCorp Vault or similar"""
    # This is a placeholder - implement based on your secret management system
    import requests
    
    vault_url = os.environ.get('VAULT_URL')
    vault_token = os.environ.get('VAULT_TOKEN')
    
    if vault_url and vault_token:
        # Fetch secrets from vault
        headers = {'X-Vault-Token': vault_token}
        response = requests.get(f'{vault_url}/v1/secret/myapp', headers=headers)
        
        if response.status_code == 200:
            secrets = response.json()['data']
            for key, value in secrets.items():
                os.environ[key.upper()] = value
            print("✓ Secrets loaded from Vault")
```

## Environment Variable Best Practices

### Configuration Hierarchy

```python
class HierarchicalConfig(ParamsProto):
    """Configuration with clear hierarchy: CLI > ENV > Config File > Defaults"""
    
    # 1. Command line arguments (highest priority)
    app_name = Proto("Application name", default="myapp")
    
    # 2. Environment variables (medium priority)  
    port = Proto("Server port", env="PORT", dtype=int, default=8000)
    
    # 3. Configuration file (lower priority)
    # This would be loaded via utils.read_deps() before parsing
    
    # 4. Defaults (lowest priority) - specified in Proto definitions

def load_hierarchical_config():
    """Load configuration with proper hierarchy"""
    # 1. Load from config file first (if exists)
    config_file = os.environ.get('CONFIG_FILE', 'config.yaml')
    if os.path.exists(config_file):
        from params_proto.v2.utils import read_deps
        file_config = read_deps(config_file)
        HierarchicalConfig._update(file_config)
    
    # 2. Environment variables override config file (automatic)
    # 3. Command line arguments override environment (automatic)
    HierarchicalConfig.parse()

# Usage examples:
# 1. Default: python app.py
# 2. Environment: PORT=9000 python app.py  
# 3. Config file: CONFIG_FILE=prod.yaml python app.py
# 4. CLI override: python app.py --HierarchicalConfig.port 3000
```

### Validation and Error Handling

```python
class ValidatedEnvConfig(ParamsProto):
    # URL validation
    database_url = Proto("Database URL", env="DATABASE_URL", default="sqlite:///dev.db")
    
    # Email validation
    admin_email = Proto("Admin email", env="ADMIN_EMAIL", default="admin@example.com")
    
    # Port range validation
    port = Proto("Server port", env="PORT", dtype=int, default=8000)
    
    # Choice validation
    log_level = Proto("Log level", env="LOG_LEVEL", default="INFO")

def validate_env_config():
    """Comprehensive validation of environment configuration"""
    ValidatedEnvConfig.parse()
    
    # URL validation
    import re
    url_pattern = re.compile(r'^(sqlite://|postgresql://|mysql://)')
    assert url_pattern.match(ValidatedEnvConfig.database_url), \
        f"Invalid database URL: {ValidatedEnvConfig.database_url}"
    
    # Email validation  
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    assert email_pattern.match(ValidatedEnvConfig.admin_email), \
        f"Invalid admin email: {ValidatedEnvConfig.admin_email}"
    
    # Port range validation
    assert 1 <= ValidatedEnvConfig.port <= 65535, \
        f"Port must be between 1-65535, got {ValidatedEnvConfig.port}"
    
    # Log level validation
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    assert ValidatedEnvConfig.log_level.upper() in valid_levels, \
        f"Log level must be one of {valid_levels}, got {ValidatedEnvConfig.log_level}"
    
    print("✓ Environment configuration validation passed")

# Safe configuration loading
try:
    validate_env_config()
except AssertionError as e:
    print(f"Configuration error: {e}")
    exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    exit(1)
```