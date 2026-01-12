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

  4. OR operation with multiple env var names (tries each in order):
      api_key: str = EnvVar @ "API_KEY" @ "SECRET_KEY" | "default"
      # Or function syntax:
      api_key: str = EnvVar("API_KEY", "SECRET_KEY", default="default")

  The pipe operator (|) allows clean chaining of env var name with fallback value.
  Values are loaded lazily - environment variables are read at access time, not definition time.
  """

  def __init__(self, *templates: str, default: Any = None, dtype: type = None):
    """
    Create an environment variable reader.

    Args:
        *templates: One or more environment variable names or template strings.
                   When multiple are provided, they are checked in order (OR operation).
        default: Default value if no environment variable is set
        dtype: Optional type to convert the value to (overrides annotation inference)
    """
    # Store templates as a tuple for immutability
    self.templates = tuple(t for t in templates if t is not None)
    self.default = default
    self.dtype = dtype
    self._env_name = None
    # For lazy loading - cached value
    self._cached_value = None
    self._is_cached = False

  @property
  def template(self):
    """Backward compatibility: return first template or None."""
    return self.templates[0] if self.templates else None

  def __matmul__(self, other: Any):
    """
    Support EnvVar @ "VAR_NAME" syntax, chainable for OR operation.

    Examples:
        EnvVar @ "VAR_NAME"              # Single env var
        EnvVar @ "VAR1" @ "VAR2"         # OR: try VAR1, then VAR2
        EnvVar @ "VAR1" @ "VAR2" | default  # With fallback

    Note: The | operator has lower precedence than @, so
    `EnvVar @ "A" @ "B" | default` is parsed as `(EnvVar @ "A" @ "B") | default`.

    Args:
        other: Either an environment variable name (str) or a default value

    Returns:
        New _EnvVar instance configured with the given parameter
    """
    if isinstance(other, str):
      # EnvVar @ "VAR_NAME" - add to templates list (OR operation)
      return _EnvVar(*self.templates, other, default=self.default, dtype=self.dtype)
    else:
      # EnvVar @ some_value - treat as default value
      return _EnvVar(*self.templates, default=other, dtype=self.dtype)

  def __or__(self, other: Any):
    """
    Support chaining with | to specify default value.

    Syntax: EnvVar @ "VAR_NAME" | default_value

    Note: Using `or` instead of `|` will NOT work because `or` has lower
    precedence than @ and evaluates truthiness instead of calling __or__.

    Args:
        other: Default value to use if env var is not set

    Returns:
        New _EnvVar instance with both template(s) and default
    """
    return _EnvVar(*self.templates, default=other, dtype=self.dtype)

  def __call__(self, *templates: str, default: Any = None, dtype: type = None):
    """
    Support EnvVar("VAR_NAME", ..., default=...) function call syntax.

    Examples:
        EnvVar("DATABASE_URL", default="localhost")
        EnvVar("API_KEY", "SECRET_KEY", default="fallback")  # OR operation
        EnvVar("PORT", dtype=int, default=8080)  # With type conversion

    Args:
        *templates: One or more environment variable names or template strings
        default: Default value if no environment variable is set
        dtype: Optional type to convert the value to

    Returns:
        New _EnvVar instance configured with the given parameters
    """
    return _EnvVar(*templates, default=default, dtype=dtype)

  def _resolve_single(self, name: str) -> tuple[Any, bool]:
    """
    Resolve a single environment variable name/template.

    Args:
        name: Environment variable name or template string

    Returns:
        Tuple of (value, found) where found indicates if the env var was set
    """
    # Handle template strings with $ prefix or ${} syntax
    if name.startswith("$") or "${" in name:
      # Parse and expand template
      vars_in_template = parse_env_template(name)
      if vars_in_template:
        # Check if any variable in template is actually set
        any_set = any(var in os.environ for var in vars_in_template)
        if not any_set:
          return None, False

        # Expansion for both $VAR and ${VAR} syntax
        expanded = name
        for var in vars_in_template:
          var_value = os.environ.get(var, "")
          # Replace both ${VAR} and $VAR forms
          expanded = expanded.replace(f"${{{var}}}", var_value)
          expanded = expanded.replace(f"${var}", var_value)
        return expanded, True

    # Simple env var lookup
    if name in os.environ:
      return os.environ[name], True
    return None, False

  def get(self, *, lazy: bool = True) -> Any:
    """
    Get the value from environment variable(s).

    When multiple templates are specified (OR operation), tries each in order
    and returns the first one that is set in the environment.

    Args:
        lazy: If True (default), cache the result for subsequent calls.
              Set to False to always re-read from environment.

    Returns:
        Value from environment or default, converted to dtype if specified
    """
    from params_proto.type_utils import _convert_type

    # Return cached value if lazy loading is enabled and we have a cached value
    if lazy and self._is_cached:
      return self._cached_value

    # No templates means return default
    if not self.templates:
      result = self.default
    else:
      # Try each template in order (OR operation)
      result = None
      for name in self.templates:
        value, found = self._resolve_single(name)
        if found:
          result = value
          break
      else:
        # No env var was found, use default
        result = self.default

    # Apply dtype conversion if specified and we have a value
    if result is not None and self.dtype is not None:
      result = _convert_type(result, self.dtype)

    # Cache the result for lazy loading
    if lazy:
      self._cached_value = result
      self._is_cached = True

    return result

  def invalidate_cache(self):
    """Clear the cached value, forcing re-read from environment on next get()."""
    self._cached_value = None
    self._is_cached = False

  def __repr__(self):
    if self.templates:
      if len(self.templates) == 1:
        return f"EnvVar({self.templates[0]!r}, default={self.default!r})"
      return f"EnvVar({', '.join(repr(t) for t in self.templates)}, default={self.default!r})"
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
