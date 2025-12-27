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
from params_proto.cli.help_gen import _generate_help_for_function, _generate_help_for_subcommand

T = TypeVar("T")
F = TypeVar("F", bound=Callable)


def _pascal_to_kebab(name: str) -> str:
  """Convert PascalCase/camelCase to kebab-case with acronym handling.

  Examples:
      Train → train
      TrainModel → train-model
      HTTPServer → http-server
      MLModel → ml-model
      DataLoader → data-loader
      parseHTMLDocument → parse-html-document
  """
  import re

  # Insert hyphens before capitals that mark word boundaries
  # Pattern 1: lowercase/digit followed by uppercase (e.g., myHTTP, model2D)
  result = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", name)

  # Pattern 2: uppercase followed by lowercase, but not at start (e.g., HTTPServer → HTTP-Server)
  result = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", result)

  # Replace underscores with hyphens
  result = result.replace("_", "-")

  # Convert to lowercase
  return result.lower()


# Global registry for singleton instances
_SINGLETONS: Dict[str, Any] = {}

# Share _SINGLETONS with help_gen module
import params_proto.cli.help_gen as help_gen
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


class _BoundProtoWrapper:
  """A bound version of ProtoWrapper for instance/class method calls.

  This is returned by ProtoWrapper.__get__ when accessed via an instance
  or class, providing proper self/cls binding.
  """

  def __init__(self, wrapper: "ProtoWrapper", bound_to):
    self._wrapper = wrapper
    self._bound_to = bound_to

  def __call__(self, *args, **kwargs):
    """Call the wrapped function with the bound object as first argument."""
    return self._wrapper._actual_func(self._bound_to, *args, **kwargs)

  def __getattr__(self, name):
    """Delegate attribute access to the underlying wrapper."""
    return getattr(self._wrapper, name)


class ProtoWrapper:
  """Wrapper for proto-decorated functions."""

  def __init__(
    self,
    func: Callable,
    is_cli: bool = False,
    is_prefix: bool = False,
    prog: str = None,
  ):
    # Detect and unwrap classmethod/staticmethod descriptors
    self._is_classmethod = isinstance(func, classmethod)
    self._is_staticmethod = isinstance(func, staticmethod)

    if self._is_classmethod or self._is_staticmethod:
      # Unwrap to get the actual function
      actual_func = func.__func__
    else:
      actual_func = func

    self._func = func  # Store original (may be descriptor)
    self._actual_func = actual_func  # Store unwrapped function for calling
    self._is_cli = is_cli
    self._is_prefix = is_prefix
    self._prog = prog  # Override script name for help generation
    self._overrides = {}
    self._name = actual_func.__name__

    # Copy function metadata
    self.__name__ = actual_func.__name__
    self.__doc__ = actual_func.__doc__
    self.__module__ = actual_func.__module__

    # Extract function signature info
    self._sig = inspect.signature(actual_func)
    self._params = {}
    self._annotations = {}
    self._defaults = {}
    self._field_docs = {}

    # Determine if we should skip the first parameter
    # - classmethod: always skip first param (cls)
    # - staticmethod: never skip (no self/cls)
    # - regular function: skip if first param is named self/cls (likely a method)
    skip_first_param = self._is_classmethod
    is_first_param = True

    for param_name, param in self._sig.parameters.items():
      # Skip **kwargs (VAR_KEYWORD)
      if param.kind == inspect.Parameter.VAR_KEYWORD:
        continue

      # Skip *args (VAR_POSITIONAL)
      if param.kind == inspect.Parameter.VAR_POSITIONAL:
        is_first_param = False
        continue

      # Skip first param for classmethod, or if it's named self/cls for regular functions
      if is_first_param:
        if skip_first_param or (not self._is_staticmethod and param_name in ("self", "cls")):
          is_first_param = False
          continue

      is_first_param = False

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

    # Initialize sweep mode state
    self._sweep_mode = False
    self._sweep_callback = None
    self._sweep_data = {}

  def _enable_sweep_mode(self, callback):
    """Enter sweep mode with a callback for recording values."""
    self._sweep_mode = True
    self._sweep_callback = callback
    self._sweep_data = {}

  def _disable_sweep_mode(self):
    """Exit sweep mode and return to normal operation."""
    self._sweep_mode = False
    self._sweep_callback = None
    self._sweep_data = {}

  def __get__(self, obj, objtype=None):
    """Descriptor protocol for proper method binding.

    This allows ProtoWrapper to work correctly when wrapping:
    - staticmethod: returns self (no binding needed)
    - classmethod: returns a bound version with cls
    - instance methods: returns a bound version with self
    """
    if self._is_staticmethod:
      # staticmethod: no binding, return the wrapper as-is
      return self
    elif self._is_classmethod:
      # classmethod: bind to the class
      if objtype is None:
        objtype = type(obj)
      return _BoundProtoWrapper(self, objtype)
    elif obj is not None:
      # Instance method: bind to the instance
      return _BoundProtoWrapper(self, obj)
    else:
      # Class-level access of regular method: return wrapper
      return self

  @property
  def _prefix(self):
    """Return the prefix for this wrapper (function name for CLI wrappers)."""
    return _pascal_to_kebab(self._name) if self._is_prefix else None

  def _update(self, __d: dict = None, **kwargs):
    """Update overrides from dict or kwargs."""
    # Ensure _overrides exists
    if not hasattr(self, "_overrides"):
      object.__setattr__(self, "_overrides", {})

    # Process dict argument
    if __d:
      prefix = self._prefix
      if prefix:
        prefix_key = f"{prefix}."
        for k, v in __d.items():
          if k.startswith(prefix_key):
            param_name = k[len(prefix_key):]
            if param_name in self._params:
              self._overrides[param_name] = v
          elif "." not in k:
            if k in self._params:
              self._overrides[k] = v
      else:
        for k, v in __d.items():
          if "." not in k:
            if k in self._params:
              self._overrides[k] = v

    # Process kwargs
    for k, v in kwargs.items():
      if k in self._params:
        self._overrides[k] = v

  def __setattr__(self, name, value):
    if name.startswith("_") or name in (
      "__name__",
      "__doc__",
      "__module__",
      "__help_str__",
    ):
      object.__setattr__(self, name, value)
    else:
      # Check if we're in sweep mode
      if hasattr(self, "_sweep_mode") and self._sweep_mode:
        # In sweep mode: validate attribute exists in params
        if name not in self._params:
          raise AttributeError(
            f"Cannot set non-existent parameter '{name}' on {self._name} during sweep. "
            f"Available parameters: {', '.join(self._params.keys())}"
          )

        # Record the value and call callback
        self._sweep_data[name] = {"_value": value}

        if self._sweep_callback:
          prefix = self._prefix
          self._sweep_callback(name, value, prefix)
      else:
        # Normal mode: store override
        if not hasattr(self, "_overrides"):
          object.__setattr__(self, "_overrides", {})
        self._overrides[name] = value

  def __getattr__(self, name):
    # Use object.__getattribute__ to avoid recursion when accessing internal attrs

    # Check if we're in sweep mode and have recorded data
    try:
      sweep_mode = object.__getattribute__(self, "_sweep_mode")
      if sweep_mode:
        sweep_data = object.__getattribute__(self, "_sweep_data")
        if name in sweep_data:
          return sweep_data[name]["_value"]
    except AttributeError:
      pass

    try:
      overrides = object.__getattribute__(self, "_overrides")
      if name in overrides:
        return overrides[name]
    except AttributeError:
      pass

    # Check bind context (from proto.bind())
    try:
      params = object.__getattribute__(self, "_params")
      prefix = object.__getattribute__(self, "_prefix")

      if prefix:
        # For @proto.prefix: check prefixed keys like "train.lr"
        prefixed_key = f"{prefix}.{name}"
        if prefixed_key in _BIND_CONTEXT:
          return _BIND_CONTEXT[prefixed_key]

      # Check for direct key (non-prefixed) - works for both @proto.cli and @proto.prefix
      if name in _BIND_CONTEXT and "." not in name:
        if name in params:
          return _BIND_CONTEXT[name]
    except AttributeError:
      pass

    # Try to get default value
    try:
      defaults = object.__getattribute__(self, "_defaults")
      if name in defaults:
        return defaults[name]
    except AttributeError:
      pass

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
          from params_proto.cli.ansi_help import colorize_help
          from params_proto.cli.cli_parse import _is_union_type, _get_union_classes, _match_class_by_name

          # Check if a subcommand precedes the help flag
          help_idx = sys.argv.index('--help') if '--help' in sys.argv else sys.argv.index('-h')
          subcommand_class = None

          # Look for Union type parameters and check if a subcommand name is in argv before --help
          for param_name, param_info in self._params.items():
            annotation = param_info["annotation"]
            if _is_union_type(annotation):
              union_classes = _get_union_classes(annotation)
              # Check each arg before --help to see if it matches a union class
              for arg in sys.argv[1:help_idx]:
                if not arg.startswith('-'):
                  matched = _match_class_by_name(arg, union_classes)
                  if matched:
                    subcommand_class = matched
                    break
              if subcommand_class:
                break

          if subcommand_class:
            # Show help for the subcommand only
            from pathlib import Path
            script_name = Path(sys.argv[0]).name if sys.argv[0] else self.__name__
            print(colorize_help(_generate_help_for_subcommand(subcommand_class, script_name)))
          else:
            # Show main help
            print(colorize_help(self.__help_str__))
          sys.exit(0)

        # Parse CLI arguments from sys.argv into kwargs
        from params_proto.cli.cli_parse import parse_cli_args
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



class ptype(type):
  """Metaclass for proto-decorated classes that intercepts attribute access."""

  @property
  def _prefix(cls):
    """Return the proto prefix for this class."""
    return type.__getattribute__(cls, "__proto_prefix__")

  def _enable_sweep_mode(cls, callback):
    """Enter sweep mode with a callback for recording values."""
    type.__setattr__(cls, "__proto_sweep_mode__", True)
    type.__setattr__(cls, "__proto_sweep_callback__", callback)
    type.__setattr__(cls, "__proto_sweep_data__", {})

  def _disable_sweep_mode(cls):
    """Exit sweep mode and return to normal operation."""
    type.__setattr__(cls, "__proto_sweep_mode__", False)
    type.__setattr__(cls, "__proto_sweep_callback__", None)
    type.__setattr__(cls, "__proto_sweep_data__", {})

  def _update(cls, __d=None, **kwargs):
    """Update overrides from dict or kwargs."""
    annotations = type.__getattribute__(cls, "__proto_annotations__")
    overrides = type.__getattribute__(cls, "__proto_overrides__")

    if __d:
      prefix = type.__getattribute__(cls, "__proto_prefix__")
      if prefix:
        prefix_key = f"{prefix}."
        for k, v in __d.items():
          if k.startswith(prefix_key):
            param_name = k[len(prefix_key):]
            if param_name in annotations:
              overrides[param_name] = v
          elif "." not in k:
            if k in annotations:
              overrides[k] = v
      else:
        for k, v in __d.items():
          if "." not in k:
            if k in annotations:
              overrides[k] = v

    for k, v in kwargs.items():
      if k in annotations:
        overrides[k] = v

  def __setattr__(cls, name, value):
    # Allow setting internal proto attributes
    if name.startswith("__proto_"):
      type.__setattr__(cls, name, value)
    elif name.startswith("_"):
      # Allow private attributes
      type.__setattr__(cls, name, value)
    else:
      # Check if we're in sweep mode
      sweep_mode = getattr(cls, "__proto_sweep_mode__", False)

      if sweep_mode:
        # In sweep mode: validate attribute exists in annotations
        annotations = type.__getattribute__(cls, "__proto_annotations__")
        if name not in annotations:
          raise AttributeError(
            f"Cannot set non-existent attribute '{name}' on {cls.__name__} during sweep. "
            f"Available attributes: {', '.join(annotations.keys())}"
          )

        # Record the value and call callback
        sweep_data = type.__getattribute__(cls, "__proto_sweep_data__")
        sweep_data[name] = {"_value": value}

        callback = type.__getattribute__(cls, "__proto_sweep_callback__")
        if callback:
          prefix = type.__getattribute__(cls, "__proto_prefix__")
          callback(name, value, prefix)
      else:
        # Normal mode: store in overrides
        if not hasattr(cls, "__proto_overrides__"):
          type.__setattr__(cls, "__proto_overrides__", {})
        cls.__proto_overrides__[name] = value

  def __getattribute__(cls, name):
    # Allow access to internal proto attributes
    if name.startswith("__proto_") or name.startswith("_"):
      return type.__getattribute__(cls, name)

    # Check if we're in sweep mode and have recorded data
    try:
      sweep_mode = type.__getattribute__(cls, "__proto_sweep_mode__")
      if sweep_mode:
        sweep_data = type.__getattribute__(cls, "__proto_sweep_data__")
        if name in sweep_data:
          return sweep_data[name]["_value"]
    except AttributeError:
      pass

    # Check overrides first
    try:
      overrides = type.__getattribute__(cls, "__proto_overrides__")
      if name in overrides:
        return overrides[name]
    except AttributeError:
      pass

    # Check bind context
    try:
      prefix = type.__getattribute__(cls, "__proto_prefix__")
      if prefix:
        # Look for prefixed keys like "config.lr"
        key = f"{prefix}.{name}"
        if key in _BIND_CONTEXT:
          return _BIND_CONTEXT[key]
      else:
        # Look for direct keys
        if name in _BIND_CONTEXT and "." not in name:
          return _BIND_CONTEXT[name]
    except AttributeError:
      pass

    # Fall back to defaults, then class attributes
    try:
      defaults = type.__getattribute__(cls, "__proto_defaults__")
      if name in defaults:
        return defaults[name]
    except AttributeError:
      pass

    return type.__getattribute__(cls, name)

  def __call__(cls, *args, **kwargs):
    """Create instance with merged values from defaults, overrides, and bind context."""
    final_kwargs = {}

    # Start with defaults
    if hasattr(cls, "__proto_defaults__"):
      final_kwargs.update(cls.__proto_defaults__)

    # Apply class-level overrides
    if hasattr(cls, "__proto_overrides__"):
      final_kwargs.update(cls.__proto_overrides__)

    # Apply bind context
    if hasattr(cls, "__proto_prefix__"):
      prefix = cls.__proto_prefix__
      if prefix:
        # Handle prefixed keys
        for key, value in _BIND_CONTEXT.items():
          if key.startswith(f"{prefix}."):
            param_name = key.split(".", 1)[1]
            final_kwargs[param_name] = value
      else:
        # Handle non-prefixed keys
        annotations = getattr(cls, "__proto_annotations__", {})
        for key, value in _BIND_CONTEXT.items():
          if "." not in key and key in annotations:
            final_kwargs[key] = value

    # Apply direct kwargs
    final_kwargs.update(kwargs)

    # Get the original class
    original_cls = type.__getattribute__(cls, "__proto_original_class__")

    # Create instance
    instance = object.__new__(original_cls)

    # Set attributes
    annotations = getattr(cls, "__proto_annotations__", {})
    for name in annotations.keys():
      if name in final_kwargs:
        setattr(instance, name, final_kwargs[name])
      elif hasattr(cls, "__proto_defaults__") and name in cls.__proto_defaults__:
        setattr(instance, name, cls.__proto_defaults__[name])
      else:
        # Required field
        setattr(instance, name, None)

    # Copy methods from original class and wrap to return self
    for name in dir(original_cls):
      # Skip dunder methods and proto fields (fields are handled above)
      if name.startswith("__") or name in annotations:
        continue

      # Check raw descriptor in MRO to detect staticmethod/classmethod (handles inheritance)
      raw_attr = None
      for klass in original_cls.__mro__:
        if name in klass.__dict__:
          raw_attr = klass.__dict__[name]
          break

      attr = getattr(original_cls, name)

      # Only process actual methods (staticmethod, classmethod, or function)
      if isinstance(raw_attr, staticmethod):
        # For staticmethod, use directly (no binding needed)
        method = attr
      elif isinstance(raw_attr, classmethod) or inspect.isfunction(raw_attr) or inspect.ismethod(attr):
        # For instance methods and classmethods, bind to instance
        # Note: classmethods bound to instance is intentional for @proto
        # semantics where instances have all attributes accessible
        method = attr.__get__(instance, original_cls)
      else:
        # Not a method (e.g., _EnvVar, property, or other callable), skip
        continue

      # Wrap it to return self if it returns None
      def make_wrapper(m):
        def wrapper(*args, **kwargs):
          result = m(*args, **kwargs)
          return instance if result is None else result

        return wrapper

      setattr(instance, name, make_wrapper(method))

    # Call __post_init__ if defined (like dataclasses)
    if hasattr(instance, '__post_init__'):
      instance.__post_init__()

    return instance


def proto(
  cls_or_func: Callable = None,
  *,
  cli: bool = False,
  prefix: bool = False,
  prefix_name: str = None,
  prog: str = None,
):
  """
  Main proto decorator that converts a class or function into a proto config object.

  Args:
      cls_or_func: The class or function to decorate
      cli: If True, this is a CLI entry point (generates help)
      prefix: If True, creates a singleton instance with prefix in CLI
      prefix_name: Custom prefix name (defaults to lowercase class/function name)
      prog: Optional program name override for help generation (useful for testing)

  Returns:
      Decorated class/function with attribute setting and calling support
  """

  def decorator(obj):
    # Handle functions, classmethod, and staticmethod descriptors
    if inspect.isfunction(obj) or isinstance(obj, (classmethod, staticmethod)):
      wrapper = ProtoWrapper(obj, is_cli=cli, is_prefix=prefix, prog=prog)
      if prefix:
        # Use custom prefix name if provided, otherwise convert to kebab-case
        func_name = obj.__func__.__name__ if isinstance(obj, (classmethod, staticmethod)) else obj.__name__
        singleton_key = prefix_name if prefix_name else _pascal_to_kebab(func_name)
        _SINGLETONS[singleton_key] = wrapper
      return wrapper
    elif inspect.isclass(obj):
      # New metaclass-based approach for classes
      # Extract annotations and defaults, including inherited ones
      # Walk MRO in reverse so child annotations override parent
      annotations = {}
      defaults = {}
      field_docs = _extract_docs_from_source(obj)

      for klass in reversed(obj.__mro__):
        if klass is object:
          continue
        klass_annotations = getattr(klass, "__annotations__", {})
        annotations.update(klass_annotations)

      for name in annotations.keys():
        if hasattr(obj, name):
          value = getattr(obj, name)
          # Check for EnvVar first (before callable check, since _EnvVar has __call__)
          is_env_var = (
            hasattr(value, "__class__") and value.__class__.__name__ == "_EnvVar"
          )

          if is_env_var:
            # Resolve env var at decoration time
            env_value = value.get()
            annotation = annotations.get(name, str)

            # Apply type conversion based on dtype (if provided) or annotation
            if env_value is not None:
              target_type = value.dtype if value.dtype is not None else annotation
              resolved_value = _convert_type(env_value, target_type)
            else:
              resolved_value = value.default  # Use the EnvVar's default value
            defaults[name] = resolved_value
          elif not callable(value):
            # Skip methods, but include regular values
            defaults[name] = value

      # Handle existing metaclass
      existing_meta = type(obj)
      if existing_meta is not type:
        # Merge with existing metaclass
        class MergedMeta(existing_meta, ptype):
          pass
        metaclass = MergedMeta
      else:
        metaclass = ptype

      # Recreate the class with ptype as its metaclass
      # Collect class namespace (attributes and methods)
      namespace = {}
      for key in dir(obj):
        if not key.startswith("__") or key in ("__annotations__", "__module__", "__qualname__", "__doc__"):
          try:
            # Use __dict__ to preserve classmethod/staticmethod descriptors
            # getattr() would return bound methods instead of descriptors
            if key in obj.__dict__:
              namespace[key] = obj.__dict__[key]
            else:
              namespace[key] = getattr(obj, key)
          except AttributeError:
            pass

      # Create new class with metaclass
      new_cls = metaclass(
        obj.__name__,
        obj.__bases__,
        namespace
      )

      # Store proto metadata on the class
      type.__setattr__(new_cls, "__proto_overrides__", {})
      type.__setattr__(new_cls, "__proto_defaults__", defaults)
      type.__setattr__(new_cls, "__proto_annotations__", annotations)
      type.__setattr__(new_cls, "__proto_field_docs__", field_docs)
      type.__setattr__(new_cls, "__proto_is_cli__", cli)
      type.__setattr__(new_cls, "__proto_is_prefix__", prefix)
      type.__setattr__(new_cls, "__proto_original_class__", obj)

      # Store prefix name (custom or kebab-case class name for prefixed configs)
      if prefix:
        # Use custom prefix name if provided, otherwise convert to kebab-case
        singleton_key = prefix_name if prefix_name else _pascal_to_kebab(obj.__name__)
        type.__setattr__(new_cls, "__proto_prefix__", singleton_key)
        _SINGLETONS[singleton_key] = new_cls
      else:
        type.__setattr__(new_cls, "__proto_prefix__", None)

      return new_cls
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


def prefix_decorator(cls_or_func=None, name: str = None):
  """Decorator for prefixed singleton configs.

  Args:
      cls_or_func: The class or function to decorate (or prefix name if called with string)
      name: Optional custom prefix name (defaults to lowercase class/function name)

  Examples:
      @proto.prefix
      class Config: ...

      @proto.prefix("custom")
      class Config: ...
  """
  # Handle @proto.prefix("custom") - cls_or_func is actually the prefix name
  if isinstance(cls_or_func, str):
    prefix_name = cls_or_func
    return lambda obj: proto(obj, prefix=True, prefix_name=prefix_name)

  # Handle @proto.prefix or @proto.prefix(name="custom")
  if cls_or_func is None:
    # Called with keyword arguments: @proto.prefix(name="custom")
    return lambda obj: proto(obj, prefix=True, prefix_name=name)
  else:
    # Called without arguments: @proto.prefix
    return proto(cls_or_func, prefix=True, prefix_name=name)


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


def partial(config_class: Type, method: bool = False):
  """
  Decorator that injects parameter defaults from a config class into a function.

  This allows you to define a plain class with type-annotated attributes and their
  defaults, then use those defaults to populate function parameters automatically.

  Args:
      config_class: A class with type-annotated attributes serving as parameter defaults
      method: If True, wraps as a method (for class methods)

  Example:
      class Config:
        lr: float = 0.01
        batch_size: int = 32

      @proto.partial(Config)
      def train() -> None:
        print(f"Learning Rate: {Config.lr}")
        print(f"Batch Size: {Config.batch_size}")

      # Supports direct attribute modification:
      Config.lr = 0.001
      train()  # Uses updated lr value

      # Supports hyperparameter sweeps:
      for Config.lr in [0.01, 0.001, 0.0001]:
        train()

  Returns:
      Decorated function with config values injected as defaults
  """
  from functools import wraps, partialmethod

  def decorator(func: Callable) -> Callable:
    # Detect and unwrap classmethod/staticmethod descriptors
    is_classmethod = isinstance(func, classmethod)
    is_staticmethod = isinstance(func, staticmethod)

    if is_classmethod or is_staticmethod:
      actual_func = func.__func__
    else:
      actual_func = func

    sig = inspect.signature(actual_func)
    params = sig.parameters

    # For classmethod, skip the first parameter (cls)
    if is_classmethod:
      params = dict(list(params.items())[1:])

    # Check for keyword-only parameters without defaults
    has_keyword_only = any(
      p.kind == inspect.Parameter.KEYWORD_ONLY and p.default == inspect.Parameter.empty
      for p in params.values()
    )

    @wraps(actual_func)
    def wrapper(*args, **kwargs):
      # Build overrides from config class
      overrides = {}

      # Determine which parameters are already bound by positional args
      # For classmethod, first arg is cls which we've excluded from params
      param_names = list(params.keys())
      effective_args_len = len(args) - 1 if is_classmethod else len(args)
      positional_bound = set(param_names[:max(0, effective_args_len)])

      for param_name, param in params.items():
        # Skip if already bound by positional argument
        if param_name in positional_bound:
          continue

        # Skip if config class doesn't have this attribute
        if not hasattr(config_class, param_name):
          continue

        # Skip if function already has a default for this parameter
        if param.default != inspect.Parameter.empty:
          continue

        # Skip positional-or-keyword params if there are keyword-only params without defaults
        if has_keyword_only and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
          continue

        # Get value from config class
        value = getattr(config_class, param_name)

        # Handle Proto objects from v2 API (for backwards compatibility)
        if hasattr(value, "default"):
          overrides[param_name] = value.default
        else:
          overrides[param_name] = value

      # Merge with explicit kwargs (kwargs take precedence)
      overrides.update(kwargs)

      # Call function with merged parameters
      return actual_func(*args, **overrides)

    # Re-wrap in classmethod/staticmethod if needed
    if is_classmethod:
      return classmethod(wrapper)
    elif is_staticmethod:
      return staticmethod(wrapper)
    elif method:
      return partialmethod(wrapper)
    return wrapper

  return decorator


proto.partial = partial


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
        Value from environment or default, converted to dtype if specified
    """
    import os

    from params_proto.parse_env_template import parse_env_template
    from params_proto.type_utils import _convert_type

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
        value = expanded if expanded != name else self.default
        # Apply dtype conversion if specified
        if value is not None and self.dtype is not None:
          return _convert_type(value, self.dtype)
        return value

    # Simple env var lookup
    value = os.environ.get(name)
    if value is not None:
      # Apply dtype conversion if specified
      if self.dtype is not None:
        return _convert_type(value, self.dtype)
      return value
    return self.default

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
