---
title: Environment Variables
description: Read configuration from environment variables with EnvVar
---

# Environment Variables

params-proto supports reading configuration from environment variables.

## Basic Usage

```python
from params_proto import proto, EnvVar

@proto.cli
def train(
    lr: float = EnvVar @ "LEARNING_RATE" | 0.001,  # From env or default
    api_key: str = EnvVar @ "API_KEY",  # Required env var
    host: str = EnvVar @ "HOST" | "localhost",
): ...
```

## Syntax

```python
EnvVar @ "ENV_VAR_NAME" | default_value
```

- `@` specifies the environment variable name
- `|` provides a fallback default value
- Without `|`, the env var is required

## Type Conversion

Values are automatically converted based on type hint:

```python
@proto.cli
def train(
    lr: float = EnvVar @ "LR" | 0.001,  # String → float
    batch_size: int = EnvVar @ "BATCH" | 32,  # String → int
    debug: bool = EnvVar @ "DEBUG" | False,  # String → bool
): ...
```

Boolean conversion:
- True: `"true"`, `"1"`, `"yes"`, `"on"` (case-insensitive)
- False: `"false"`, `"0"`, `"no"`, `"off"`

## Required Environment Variables

```python
@proto.cli
def deploy(
    api_key: str = EnvVar @ "API_KEY",  # No default = required
): ...
```

Missing required env var raises error at import time.

## With @proto.prefix

```python
@proto.prefix
class Config:
    host: str = EnvVar @ "HOST" | "localhost"
    port: int = EnvVar @ "PORT" | 8080
    debug: bool = EnvVar @ "DEBUG" | False
```

## With Inheritance

EnvVar fields are inherited and type-converted correctly:

```python
class BaseConfig:
    host: str = EnvVar @ "HOST" | "localhost"
    port: int = EnvVar @ "PORT" | 8080
    debug: bool = EnvVar @ "DEBUG" | False

@proto.prefix
class AppConfig(BaseConfig):
    timeout: int = EnvVar @ "TIMEOUT" | 30
    api_key: str = EnvVar @ "API_KEY"
```

```bash
HOST=10.0.0.1 PORT=3000 DEBUG=true TIMEOUT=60 API_KEY=secret python app.py
```

Both inherited and child EnvVar fields:
- Resolve from environment variables
- Convert to the correct type (str, int, bool, float)
- Use fallback defaults when env var is not set

## Template Expansion

Multiple variables in one value:

```python
@proto.cli
def connect(
    url: str = EnvVar @ "PROTOCOL://$HOST:$PORT/api",
): ...
```

```bash
PROTOCOL=https HOST=example.com PORT=443 python connect.py
# url = "https://example.com:443/api"
```

Supported syntax:
- `$VAR`
- `${VAR}`

## Priority Order

1. CLI arguments (highest)
2. Direct assignment
3. Environment variables
4. Default values (lowest)

```python
@proto.cli
def train(
    lr: float = EnvVar @ "LR" | 0.001,
): ...
```

```bash
# Uses env var
LR=0.01 python train.py
# lr = 0.01

# CLI overrides env var
LR=0.01 python train.py --lr 0.1
# lr = 0.1
```

## In Help Text

Environment variables appear in help:

```
--lr FLOAT    Learning rate (default: $LEARNING_RATE or 0.001)
```

## Common Patterns

### Database Configuration

```python
@proto.prefix
class Database:
    host: str = EnvVar @ "DB_HOST" | "localhost"
    port: int = EnvVar @ "DB_PORT" | 5432
    user: str = EnvVar @ "DB_USER" | "postgres"
    password: str = EnvVar @ "DB_PASSWORD"  # Required, no default
    name: str = EnvVar @ "DB_NAME" | "myapp"
```

### API Keys

```python
@proto.prefix
class API:
    openai_key: str = EnvVar @ "OPENAI_API_KEY"
    anthropic_key: str = EnvVar @ "ANTHROPIC_API_KEY"
```

### Feature Flags

```python
@proto.prefix
class Features:
    enable_cache: bool = EnvVar @ "ENABLE_CACHE" | True
    debug_mode: bool = EnvVar @ "DEBUG" | False
    log_level: str = EnvVar @ "LOG_LEVEL" | "INFO"
```

## Best Practices

1. **Use uppercase for env vars** - `LEARNING_RATE` not `learning_rate`
2. **Provide defaults for optional** - `EnvVar @ "VAR" | default`
3. **Document required vars** - In comments or README
4. **Use prefixes for grouping** - `DB_HOST`, `DB_PORT`, `API_KEY`
5. **Don't commit secrets** - Use `.env` files, not code defaults
