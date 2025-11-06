"""
Environment variable support for params-proto.

Provides EnvVar class for reading configuration from environment variables
with automatic type conversion and template expansion.
"""

import os
from typing import Any, Callable

from params_proto.parse_env_template import parse_env_template


class _EnvVar:
  """
  Environment variable reader that supports three syntaxes:

  1. Matmul operator with env var name:
      batch_size: int = EnvVar @ "BATCH_SIZE"

  2. Matmul operator with pipe for default:
      learning_rate: float = EnvVar @ "LR" | 0.001

  3. Function call syntax:
      db_url: str = EnvVar("DATABASE_URL", default="localhost")
      data_dir: str = EnvVar("$DATA_DIR/models", default="/tmp/models")

  The pipe operator (|) allows clean chaining of env var name with fallback value.
  """

  def __init__(self, template: str = None, *, default: Any = None):
    """
    Create an environment variable reader.

    Args:
        template: Environment variable name or template string (e.g., "$VAR" or "VAR")
        default: Default value if environment variable is not set
    """
    self.template = template
    self.default = default
    self._env_name = None

  def __matmul__(self, other: Any):
    """
    Support EnvVar @ "VAR_NAME" or EnvVar @ default_value syntax.

    Args:
        other: Either an environment variable name (str) or a default value

    Returns:
        New _EnvVar instance configured with the given parameter
    """
    if isinstance(other, str):
      # EnvVar @ "VAR_NAME" - treat as env var name
      return _EnvVar(template=other, default=None)
    else:
      # EnvVar @ some_value - treat as default value
      return _EnvVar(template=None, default=other)

  def __or__(self, other: Any):
    """
    Support chaining with | to specify default value.

    Syntax: EnvVar @ "VAR_NAME" | default_value

    Args:
        other: Default value to use if env var is not set

    Returns:
        New _EnvVar instance with both template and default
    """
    return _EnvVar(template=self.template, default=other)

  def __call__(self, template: str, *, default: Any = None):
    """
    Support EnvVar("VAR_NAME", default=...) function call syntax.

    Args:
        template: Environment variable name or template string
        default: Default value if environment variable is not set

    Returns:
        New _EnvVar instance configured with the given parameters
    """
    return _EnvVar(template=template, default=default)

  def get(self) -> Any:
    """
    Get the value from environment variable.

    Returns:
        Value from environment or default
    """
    # Use only the explicitly set template
    # NO auto-inference for security reasons
    if not self.template:
      return self.default

    name = self.template

    # Handle template strings with $ prefix or ${} syntax
    if name.startswith("$") or "${" in name:
      # Parse and expand template
      vars_in_template = parse_env_template(name)
      if vars_in_template:
        # Expansion for both $VAR and ${VAR} syntax
        expanded = name
        for var in vars_in_template:
          var_value = os.environ.get(var, "")
          # Replace both ${VAR} and $VAR forms
          expanded = expanded.replace(f"${{{var}}}", var_value)
          expanded = expanded.replace(f"${var}", var_value)
        return expanded if expanded != name else self.default

    # Simple env var lookup
    return os.environ.get(name, self.default)

  def __repr__(self):
    if self.template:
      return f"EnvVar({self.template!r}, default={self.default!r})"
    return f"EnvVar(default={self.default!r})"


# Create singleton instance for @ syntax, but also keep class available
EnvVar = _EnvVar()


def get_var(default: Any = None, *, env: str = None):
  """
  Mark a field as an environment variable.

  Args:
      default: Default value if env var not set
      env: Environment variable name (defaults to field name)
  """

  def getter():
    env_name = env or "FIELD_NAME"  # Will be replaced at decoration time
    return os.environ.get(env_name, default)

  return getter


def Field(fn: Callable):
  """
  Decorator to mark a method as a computed field.

  Args:
      fn: The method to decorate

  Returns:
      The decorated method
  """
  fn._is_field = True
  return fn
