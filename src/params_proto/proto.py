"""
params-proto v3 API

Core decorators and functionality for declarative parameter management.
"""

import inspect
from typing import (
  Any,
  Callable,
  Dict,
  Type,
  TypeVar,
  get_origin,
  overload,
)

# Import utilities from separate modules
from params_proto.type_utils import _convert_type
from params_proto.documentation import _extract_docs_from_source
from params_proto.help_gen import _generate_help_for_function, _generate_help_for_class

T = TypeVar("T")
F = TypeVar("F", bound=Callable)

# Global registry for singleton instances
_SINGLETONS: Dict[str, Any] = {}

# Share _SINGLETONS with help_gen module
import params_proto.help_gen as help_gen
help_gen._SINGLETONS = _SINGLETONS

# Global binding context for proto.bind()
_BIND_CONTEXT: Dict[str, Any] = {}
_BIND_STACK: list = []


class ProtoResult:
  """Result object that provides attribute access to function results."""

  def __init__(self, data: dict):
    self._data = data
    for k, v in data.items():
      setattr(self, k, v)

  def __getitem__(self, key):
    return self._data[key]

  def __repr__(self):
    return f"ProtoResult({self._data!r})"


class ProtoWrapper:
  """Wrapper for proto-decorated functions."""

  def __init__(
    self,
    func: Callable,
    is_cli: bool = False,
    is_prefix: bool = False,
    prog: str = None,
  ):
    self._func = func
    self._is_cli = is_cli
    self._is_prefix = is_prefix
    self._prog = prog  # Override script name for help generation
    self._overrides = {}
    self._name = func.__name__

    # Copy function metadata
    self.__name__ = func.__name__
    self.__doc__ = func.__doc__
    self.__module__ = func.__module__

    # Extract function signature info
    self._sig = inspect.signature(func)
    self._params = {}
    self._annotations = {}
    self._defaults = {}
    self._field_docs = {}

    for param_name, param in self._sig.parameters.items():
      if param_name == "kwargs":
        continue

      annotation = (
        param.annotation if param.annotation != inspect.Parameter.empty else str
      )
      default = param.default if param.default != inspect.Parameter.empty else None

      # Resolve EnvVar instances at decoration time
      # The environment variables are read from the declaration environment
      is_env_var = (
        hasattr(default, "__class__") and default.__class__.__name__ == "_EnvVar"
      )

      if is_env_var:
        # Resolve env var at decoration time
        # NO auto-inference from parameter name for security reasons
        env_value = default.get()

        # Apply type conversion based on dtype (if provided) or annotation
        if env_value is not None:
          # Use explicit dtype if provided, otherwise infer from annotation
          target_type = default.dtype if default.dtype is not None else annotation
          resolved_default = _convert_type(env_value, target_type)
        else:
          resolved_default = default.default  # Use the EnvVar's default value
      else:
        resolved_default = default

      self._params[param_name] = {
        "annotation": annotation,
        "default": resolved_default,
        "required": param.default == inspect.Parameter.empty,
      }
      self._annotations[param_name] = annotation
      if resolved_default is not None or param.default != inspect.Parameter.empty:
        self._defaults[param_name] = resolved_default

    # Extract documentation from source
    self._field_docs = _extract_docs_from_source(func)

    # Generate help string if CLI
    if is_cli:
      self.__help_str__ = _generate_help_for_function(self)

  def __setattr__(self, name, value):
    if name.startswith("_") or name in (
      "__name__",
      "__doc__",
      "__module__",
      "__help_str__",
    ):
      object.__setattr__(self, name, value)
    else:
      # Store override
      if not hasattr(self, "_overrides"):
        object.__setattr__(self, "_overrides", {})
      self._overrides[name] = value

  def __getattr__(self, name):
    if name in self._overrides:
      return self._overrides[name]
    raise AttributeError(
      f"'{self.__class__.__name__}' object has no attribute '{name}'"
    )

  def __call__(self, *args, **kwargs):
    # If this is a CLI entry point and no args/kwargs provided, parse CLI args
    # Only parse CLI if we're actually being run as a script (not from tests)
    if self._is_cli and not args and not kwargs:
      import sys
      import os

      # Check if we should parse CLI args
      # Don't parse if running under pytest, unittest, or other test runners
      argv0 = os.path.basename(sys.argv[0]) if sys.argv else ""
      is_test_runner = any(runner in argv0 for runner in ['pytest', 'py.test', 'unittest', 'nose', 'tox'])

      if not is_test_runner:
        # Check for help flag first - block execution if present
        if '--help' in sys.argv or '-h' in sys.argv:
          # Import ANSI colorization
          from params_proto.ansi_help import colorize_help
          print(colorize_help(self.__help_str__))
          sys.exit(0)

        # Parse CLI arguments from sys.argv into kwargs
        from params_proto.cli_parse import parse_cli_args
        kwargs = parse_cli_args(self)

    # Build final kwargs by merging:
    # 1. Defaults from function signature
    # 2. Stored overrides (from attr assignment)
    # 3. Bind context (from proto.bind())
    # 4. Direct kwargs

    final_kwargs = {}

    # Start with defaults
    for param_name, param_info in self._params.items():
      if not param_info["required"]:
        final_kwargs[param_name] = param_info["default"]

    # Apply stored overrides
    final_kwargs.update(self._overrides)

    # Apply bind context - check for both direct keys and prefixed keys
    for key, value in _BIND_CONTEXT.items():
      if key == self._name:
        # Handle case like bind(train={"lr": 0.01})
        if isinstance(value, dict):
          final_kwargs.update(value)
      elif "." not in key:
        # Direct parameter
        if key in self._params:
          final_kwargs[key] = value
      else:
        # Prefixed parameter like "train.lr"
        prefix, param = key.split(".", 1)
        if prefix == self._name and param in self._params:
          final_kwargs[param] = value

    # Apply direct kwargs (highest priority)
    # Handle special case where kwargs might contain nested config for prefixed functions
    temp_bindings = {}
    for key, value in kwargs.items():
      if key in self._params:
        final_kwargs[key] = value
      else:
        # Check if it's a prefixed function override
        if key in _SINGLETONS:
          # Store in bind context for nested calls
          temp_bindings[key] = value

    # Handle positional args
    param_names = list(self._params.keys())
    for i, arg in enumerate(args):
      if i < len(param_names):
        final_kwargs[param_names[i]] = arg

    # Temporarily add bindings for nested calls
    prev_bindings = {}
    for key, value in temp_bindings.items():
      prev_bindings[key] = _BIND_CONTEXT.get(key)
      _BIND_CONTEXT[key] = value

    try:
      # Call the original function
      result = self._func(**final_kwargs)

      # If function returns a dict, wrap in ProtoResult for attribute access
      if isinstance(result, dict):
        return ProtoResult(result)

      return result
    finally:
      # Clean up temporary bindings
      for key in temp_bindings:
        if prev_bindings[key] is None:
          _BIND_CONTEXT.pop(key, None)
        else:
          _BIND_CONTEXT[key] = prev_bindings[key]


class ProtoClass:
  """Wrapper for proto-decorated classes."""

  def __init__(self, cls: Type[T], is_cli: bool = False, is_prefix: bool = False):
    self._cls = cls
    self._is_cli = is_cli
    self._is_prefix = is_prefix
    self._overrides = {}

    # Extract annotations and defaults
    self._annotations = getattr(cls, "__annotations__", {})
    self._defaults = {}
    self._field_docs = _extract_docs_from_source(cls)

    for name in self._annotations.keys():
      if hasattr(cls, name):
        value = getattr(cls, name)
        # Skip methods
        if not callable(value):
          self._defaults[name] = value

    # Generate help if needed
    if is_cli:
      self.__help_str__ = _generate_help_for_class(self)

  def __call__(self, *args, **kwargs):
    # Create instance with merged values
    final_kwargs = {}

    # Start with defaults
    final_kwargs.update(self._defaults)

    # Apply class-level overrides
    final_kwargs.update(self._overrides)

    # Apply bind context
    for key, value in _BIND_CONTEXT.items():
      if "." not in key and key in self._annotations:
        final_kwargs[key] = value

    # Apply direct kwargs
    final_kwargs.update(kwargs)

    # Create instance
    instance = object.__new__(self._cls)

    # Set attributes
    for name in self._annotations.keys():
      if name in final_kwargs:
        setattr(instance, name, final_kwargs[name])
      elif name in self._defaults:
        setattr(instance, name, self._defaults[name])
      else:
        # Required field
        setattr(instance, name, None)

    # Copy methods from original class and wrap to return self
    for name in dir(self._cls):
      if not name.startswith("__"):
        attr = getattr(self._cls, name)
        if callable(attr):
          # Get the bound method
          bound_method = attr.__get__(instance, self._cls)

          # Wrap it to return self if it returns None
          def make_wrapper(method):
            def wrapper(*args, **kwargs):
              result = method(*args, **kwargs)
              return instance if result is None else result

            return wrapper

          setattr(instance, name, make_wrapper(bound_method))

    return instance

  def __setattr__(self, name, value):
    if name.startswith("_") or name in ("__help_str__",):
      object.__setattr__(self, name, value)
    else:
      # Store override at class level
      if not hasattr(self, "_overrides"):
        object.__setattr__(self, "_overrides", {})
      self._overrides[name] = value

  def __getattr__(self, name):
    if hasattr(self, "_overrides") and name in self._overrides:
      return self._overrides[name]
    if hasattr(self, "_cls"):
      attr = getattr(self._cls, name)
      return attr
    raise AttributeError(
      f"'{self.__class__.__name__}' object has no attribute '{name}'"
    )


def proto(
  cls_or_func: Callable = None,
  *,
  cli: bool = False,
  prefix: bool = False,
  prog: str = None,
):
  """
  Main proto decorator that converts a class or function into a proto config object.

  Args:
      cls_or_func: The class or function to decorate
      cli: If True, this is a CLI entry point (generates help)
      prefix: If True, creates a singleton instance with prefix in CLI
      prog: Optional program name override for help generation (useful for testing)

  Returns:
      Decorated class/function with attribute setting and calling support
  """

  def decorator(obj):
    if inspect.isfunction(obj):
      wrapper = ProtoWrapper(obj, is_cli=cli, is_prefix=prefix, prog=prog)
      if prefix:
        _SINGLETONS[obj.__name__] = wrapper
      return wrapper
    elif inspect.isclass(obj):
      wrapper = ProtoClass(obj, is_cli=cli, is_prefix=prefix)
      if prefix:
        _SINGLETONS[obj.__name__] = wrapper
      return wrapper
    else:
      # Handle Union types
      return obj

  if cls_or_func is None:
    # Called with arguments: @proto(cli=True, prog="train.py")
    return decorator
  else:
    # Called without arguments: @proto
    return decorator(cls_or_func)


# Convenience decorators
def cli_decorator(cls_or_func):
  """Decorator for CLI entry points."""
  return proto(cls_or_func, cli=True)


def prefix_decorator(cls_or_func):
  """Decorator for prefixed singleton configs."""
  return proto(cls_or_func, prefix=True)


proto.cli = cli_decorator
proto.prefix = prefix_decorator


class BindContext:
  """Context manager for parameter bindings that also works as a direct call."""

  def __init__(self, prev_state, **kwargs):
    self.kwargs = kwargs
    self.prev_state = prev_state

  def __enter__(self):
    # Bindings already applied in bind()
    return self

  def __exit__(self, *args):
    # Restore previous state
    _BIND_CONTEXT.clear()
    _BIND_CONTEXT.update(self.prev_state)


def bind(**kwargs):
  """
  Bind parameter overrides.

  Can be used as context manager:
      with proto.bind(seed=42, **{"train.lr": 0.01}):
          result = main()

  Or as direct call (sets global bindings):
      proto.bind(seed=42, **{"train.lr": 0.01})
      result = main()
  """
  # Save previous state before updating
  prev_state = _BIND_CONTEXT.copy()
  # Update global context directly
  _BIND_CONTEXT.update(kwargs)
  # Return context manager for optional 'with' usage (with saved state)
  return BindContext(prev_state, **kwargs)


proto.bind = bind


def parse(func: Callable, **kwargs):
  """
  Parse overrides and call a function.

  Args:
      func: The function to call
      **kwargs: Override values (can use dot notation)

  Returns:
      Result of calling func with overrides applied
  """
  with bind(**kwargs):
    return func()


proto.parse = parse


@overload
def cli(obj: F, *, prog: str = None) -> F:
  """Decorate a function/class as CLI entry point (preserves type)."""
  ...


@overload
def cli(obj: None = None, *, prog: str = None) -> Callable[[F], F]:
  """Return a decorator for CLI entry points (preserves type)."""
  ...


def cli(obj: Any = None, *, prog: str = None):
  """
  Set up an object as a CLI entry point.

  Args:
      obj: The class, function, or Union type to setup as CLI.
           If None, returns a decorator.
      prog: Optional program name override for help generation (useful for testing)

  Returns:
      The object with CLI capabilities, or a decorator if obj is None
  """
  if obj is None:
    # Called with arguments: @proto.cli(prog="train.py")
    return lambda f: proto(f, cli=True, prog=prog)

  # Handle Union types
  origin = get_origin(obj)
  if origin is not None:
    # This is a Union or other generic type
    # For Union[A, B], we don't decorate it, just return it
    return obj

  return proto(obj, cli=True, prog=prog)


proto.cli = cli


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
      port: int = EnvVar("PORT", dtype=int, default=8080)

  The pipe operator (|) allows clean chaining of env var name with fallback value.
  """

  def __init__(self, template: str = None, *, default: Any = None, dtype: type = None):
    """
    Create an environment variable reader.

    Args:
        template: Environment variable name or template string (e.g., "$VAR" or "VAR")
        default: Default value if environment variable is not set
        dtype: Optional type to convert the value to (overrides annotation inference)
    """
    self.template = template
    self.default = default
    self.dtype = dtype
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
    return _EnvVar(template=self.template, default=other, dtype=self.dtype)

  def __call__(self, template: str, *, default: Any = None, dtype: type = None):
    """
    Support EnvVar("VAR_NAME", default=..., dtype=...) function call syntax.

    Args:
        template: Environment variable name or template string
        default: Default value if environment variable is not set
        dtype: Optional type to convert the value to

    Returns:
        New _EnvVar instance configured with the given parameters
    """
    return _EnvVar(template=template, default=default, dtype=dtype)

  def get(self) -> Any:
    """
    Get the value from environment variable.

    Returns:
        Value from environment or default
    """
    import os

    from params_proto.parse_env_template import parse_env_template

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
  import os

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
