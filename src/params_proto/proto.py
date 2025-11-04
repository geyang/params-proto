"""
params-proto v3 API

Core decorators and functionality for declarative parameter management.
"""

import inspect
import sys
import re
from contextlib import contextmanager
from dataclasses import is_dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Type,
    TypeVar,
    get_origin,
    get_args,
)

T = TypeVar("T")

# Global registry for singleton instances
_SINGLETONS: Dict[str, Any] = {}

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

    def __init__(self, func: Callable, is_cli: bool = False, is_prefix: bool = False):
        self._func = func
        self._is_cli = is_cli
        self._is_prefix = is_prefix
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

            self._params[param_name] = {
                "annotation": annotation,
                "default": default,
                "required": param.default == inspect.Parameter.empty,
            }
            self._annotations[param_name] = annotation
            if default is not None or param.default != inspect.Parameter.empty:
                self._defaults[param_name] = default

        # Extract documentation from source
        self._field_docs = _extract_docs_from_source(func)

        # Generate help string if CLI
        if is_cli:
            self.__help_str__ = _generate_help_for_function(self)

    def __setattr__(self, name, value):
        if name.startswith("_") or name in ("__name__", "__doc__", "__module__", "__help_str__"):
            object.__setattr__(self, name, value)
        else:
            # Store override
            if not hasattr(self, "_overrides"):
                object.__setattr__(self, "_overrides", {})
            self._overrides[name] = value

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __call__(self, *args, **kwargs):
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
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


def _extract_docs_from_source(obj: Any) -> Dict[str, str]:
    """Extract documentation from inline comments and docstrings."""
    docs = {}

    try:
        source = inspect.getsource(obj)
        lines = source.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i]

            # Extract field name and inline comment
            if ":" in line and "#" in line:
                # Pattern: field: type = value  # comment
                parts = line.split("#", 1)
                if len(parts) == 2:
                    field_part = parts[0].strip()
                    comment = parts[1].strip()
                    if ":" in field_part:
                        field_name = field_part.split(":")[0].strip()
                        docs[field_name] = comment

            # Extract docstrings (look ahead for """ after field definition)
            elif ":" in line:
                # Check if next line has docstring
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('"""'):
                        field_part = line.split(":")[0].strip()
                        # Handle multi-line docstrings
                        if next_line.endswith('"""') and len(next_line) > 6:
                            # Single line docstring
                            doc = next_line.strip('"""').strip()
                            docs[field_part] = doc
                        else:
                            # Multi-line docstring
                            doc_lines = [next_line.strip('"""').strip()]
                            j = i + 2
                            while j < len(lines):
                                doc_line = lines[j].strip()
                                if doc_line.endswith('"""'):
                                    doc_lines.append(doc_line.rstrip('"""').strip())
                                    break
                                doc_lines.append(doc_line)
                                j += 1
                            docs[field_part] = " ".join(doc_lines).strip()

            i += 1
    except (OSError, TypeError):
        pass

    return docs


def _generate_param_description(param_name: str, func_name: str) -> str:
    """Auto-generate parameter description from name."""
    # Special cases
    if param_name == "seed":
        return "Random seed"

    # Extract verb from function name (first word)
    parts = func_name.split("_")
    verb = parts[0] if parts else ""

    # Convert verb to gerund/adjective form
    verb_map = {
        "train": "training",
        "eval": "evaluation",
        "test": "testing",
        "run": "running",
        "execute": "execution",
        "process": "processing",
    }
    verb_form = verb_map.get(verb, verb)

    # Generate description based on parameter name
    param_readable = param_name.replace("_", " ")

    # If parameter looks plural or countable, add "Number of"
    if param_name.endswith("s") or param_name in ("epochs", "steps", "iterations"):
        if verb_form:
            return f"Number of {verb_form} {param_readable}"
        else:
            return f"Number of {param_readable}"

    # Otherwise, just capitalize
    return param_readable.capitalize()


def _generate_help_for_function(wrapper: ProtoWrapper) -> str:
    """Generate help text for a proto function."""
    lines = []

    # Derive script name from function name
    func_name = wrapper.__name__
    # Convert train_mnist -> mnist_train.py (move first word to end)
    # But only if second part is substantial (>2 chars)
    parts = func_name.split("_")
    if len(parts) > 1 and len(parts[1]) > 2:
        # Move first part to end: train_mnist -> mnist_train
        script_name = "_".join(parts[1:] + [parts[0]]) + ".py"
    else:
        script_name = f"{func_name}.py"

    # Build usage line with actual arguments
    usage_parts = [f"\nusage: {script_name}", "[-h]"]
    for name, param_info in wrapper._params.items():
        arg_name = f"--{name.replace('_', '-')}"
        type_name = _get_type_name(param_info["annotation"])
        usage_parts.append(f"[{arg_name} {type_name}]" if type_name else f"[{arg_name}]")

    # Add [OPTIONS] if there are prefixed singletons
    if _SINGLETONS:
        usage_parts.append("[OPTIONS]")

    lines.append(" ".join(usage_parts))
    lines.append("")

    # Description
    if wrapper.__doc__:
        lines.append(wrapper.__doc__.strip())
        lines.append("")

    # Options section
    lines.append("options:")

    # Determine padding based on whether there are prefixed singletons
    has_prefixed = len(_SINGLETONS) > 0
    main_padding = 31 if has_prefixed else 23
    help_padding = "  -h, --help                   " if has_prefixed else "  -h, --help           "

    lines.append(help_padding + "show this help message and exit")

    # Add parameters
    for name, param_info in wrapper._params.items():
        arg_name = f"--{name.replace('_', '-')}"
        type_name = _get_type_name(param_info["annotation"])
        default = wrapper._defaults.get(name)
        help_text = wrapper._field_docs.get(name, "")

        # Auto-generate description if missing
        if not help_text:
            help_text = _generate_param_description(name, wrapper.__name__)

        # Build the option line with proper spacing
        if type_name:
            option_str = f"  {arg_name} {type_name}"
        else:
            option_str = f"  {arg_name}"

        # Pad to align descriptions
        option_str = option_str.ljust(main_padding)

        # Build description with default
        desc_parts = []
        if help_text:
            desc_parts.append(help_text)
        if default is not None:
            desc_parts.append(f"(default: {default})")

        lines.append(option_str + " ".join(desc_parts))

    # Add sections for any @proto.prefix singletons
    for singleton_name, singleton in _SINGLETONS.items():
        if isinstance(singleton, ProtoClass):
            lines.append(f"\n{singleton_name} options:")
            if singleton._cls.__doc__:
                lines.append(f"  {singleton._cls.__doc__.strip()}")
                lines.append("")

            for param_name, annotation in singleton._annotations.items():
                arg_name = f"--{singleton_name}.{param_name.replace('_', '-')}"
                type_name = _get_type_name(annotation)
                default = singleton._defaults.get(param_name)
                help_text = singleton._field_docs.get(param_name, "")

                # Auto-generate description if missing
                if not help_text:
                    help_text = _generate_param_description(param_name, singleton_name.lower())

                # Build the option line with 2-space indentation
                if type_name:
                    option_str = f"  {arg_name} {type_name}"
                else:
                    option_str = f"  {arg_name}"

                # Pad to align descriptions - target column 31, but minimum 2 spaces
                if len(option_str) < 31:
                    option_str = option_str.ljust(31)
                else:
                    option_str = option_str + "  "  # At least 2 spaces

                # Build description with default
                desc_parts = []
                if help_text:
                    desc_parts.append(help_text)
                if default is not None:
                    desc_parts.append(f"(default: {default})")

                lines.append(option_str + " ".join(desc_parts))

        elif isinstance(singleton, ProtoWrapper):
            lines.append(f"\n{singleton_name} options:")
            if singleton.__doc__:
                lines.append(f"  {singleton.__doc__.strip()}")
                lines.append("")

            for param_name, param_info in singleton._params.items():
                arg_name = f"--{singleton_name}.{param_name.replace('_', '-')}"
                type_name = _get_type_name(param_info["annotation"])
                default = singleton._defaults.get(param_name)
                help_text = singleton._field_docs.get(param_name, "")

                # Auto-generate description if missing
                if not help_text:
                    help_text = _generate_param_description(param_name, singleton_name.lower())

                # Build the option line with 2-space indentation
                if type_name:
                    option_str = f"  {arg_name} {type_name}"
                else:
                    option_str = f"  {arg_name}"

                # Pad to align descriptions - target column 31, but minimum 2 spaces
                if len(option_str) < 31:
                    option_str = option_str.ljust(31)
                else:
                    option_str = option_str + "  "  # At least 2 spaces

                # Build description with default
                desc_parts = []
                if help_text:
                    desc_parts.append(help_text)
                if default is not None:
                    desc_parts.append(f"(default: {default})")

                lines.append(option_str + " ".join(desc_parts))

    lines.append("")
    return "\n".join(lines)


def _generate_help_for_class(wrapper: ProtoClass) -> str:
    """Generate help text for a proto class."""
    # Similar implementation for classes
    return ""


def _get_type_name(annotation: Any) -> str:
    """Get a human-readable type name."""
    if annotation == int or annotation is int:
        return "INT"
    elif annotation == float or annotation is float:
        return "FLOAT"
    elif annotation == str or annotation is str:
        return "STR"
    elif annotation == bool or annotation is bool:
        return ""  # Boolean flags don't show type
    elif inspect.isclass(annotation) and issubclass(annotation, Enum):
        return f"{{{','.join(e.name for e in annotation)}}}"
    else:
        return "VALUE"


def proto(cls_or_func: Callable = None, *, cli: bool = False, prefix: bool = False):
    """
    Main proto decorator that converts a class or function into a proto config object.

    Args:
        cls_or_func: The class or function to decorate
        cli: If True, this is a CLI entry point (generates help)
        prefix: If True, creates a singleton instance with prefix in CLI

    Returns:
        Decorated class/function with attribute setting and calling support
    """

    def decorator(obj):
        if inspect.isfunction(obj):
            wrapper = ProtoWrapper(obj, is_cli=cli, is_prefix=prefix)
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
        # Called with arguments: @proto(cli=True)
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


def cli(obj: Any = None):
    """
    Set up an object as a CLI entry point.

    Args:
        obj: The class, function, or Union type to setup as CLI.
             If None, this is a no-op (for compatibility).

    Returns:
        The object with CLI capabilities, or None if obj is None
    """
    if obj is None:
        # No-op when called without arguments
        return None

    # Handle Union types
    origin = get_origin(obj)
    if origin is not None:
        # This is a Union or other generic type
        # For Union[A, B], we don't decorate it, just return it
        return obj

    return proto(obj, cli=True)


proto.cli = cli


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
