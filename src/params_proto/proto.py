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

    def __init__(self, func: Callable, is_cli: bool = False, is_prefix: bool = False, prog: str = None):
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
                hasattr(default, "__class__")
                and default.__class__.__name__ == "_EnvVar"
            )

            if is_env_var:
                # Resolve env var at decoration time
                # NO auto-inference from parameter name for security reasons
                env_value = default.get()

                # Apply type conversion based on annotation
                if env_value is not None:
                    resolved_default = _convert_type(env_value, annotation)
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
    """Extract documentation from inline comments and docstrings.

    Combines inline comments with docstring Args section. If both exist,
    they are concatenated with ". " separator (inline first, then docstring).
    """
    import re
    inline_docs = {}

    try:
        source = inspect.getsource(obj)
        lines = source.split("\n")

        # Pattern to match parameter definitions: identifier: type [= value]
        # Must have a type annotation after the colon (not just a colon followed by nothing or newline)
        param_pattern = re.compile(r'^\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*\w')

        # Docstring section headers to skip
        docstring_sections = {'Args', 'Returns', 'Raises', 'Yields', 'Examples', 'Example',
                             'Attributes', 'Note', 'Notes', 'Warning', 'Warnings', 'See'}

        # Track whether we're inside a docstring
        inside_docstring = False

        i = 0
        while i < len(lines):
            line = lines[i]

            # Track docstring boundaries
            if '"""' in line or "'''" in line:
                # Count quotes to determine if we're entering or exiting
                quote_count = line.count('"""') + line.count("'''")
                if quote_count % 2 == 1:  # Odd number means state change
                    inside_docstring = not inside_docstring
                # If both opening and closing on same line, we're not inside
                elif quote_count == 2:
                    inside_docstring = False

            # Skip lines inside docstrings
            if inside_docstring:
                i += 1
                continue

            # Check if this looks like a parameter definition
            param_match = param_pattern.match(line)
            if not param_match:
                i += 1
                continue

            field_name = param_match.group(1)

            # Skip docstring section headers (shouldn't reach here if inside docstring, but just in case)
            if field_name in docstring_sections:
                i += 1
                continue

            # Extract field name and inline comment
            if "#" in line:
                # Pattern: field: type = value  # comment
                parts = line.split("#", 1)
                if len(parts) == 2:
                    comment = parts[1].strip()
                    inline_docs[field_name] = comment

            # Extract docstrings (look ahead for """ after field definition)
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('"""'):
                    # Handle multi-line docstrings
                    if next_line.endswith('"""') and len(next_line) > 6:
                        # Single line docstring
                        doc = next_line.strip('"""').strip()
                        inline_docs[field_name] = doc
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
                        inline_docs[field_name] = " ".join(doc_lines).strip()

            i += 1
    except (OSError, TypeError):
        pass

    # Extract Args from docstring
    docstring_args = {}
    if hasattr(obj, '__doc__') and obj.__doc__:
        docstring_args = _extract_args_from_docstring(obj.__doc__)

    # Combine both sources: inline comments first, then docstring Args
    combined_docs = {}
    all_params = set(inline_docs.keys()) | set(docstring_args.keys())

    for param in all_params:
        inline = inline_docs.get(param, "")
        docstring = docstring_args.get(param, "")

        if inline and docstring:
            # Both exist: check if they're the same to avoid duplication
            if inline.strip() == docstring.strip():
                # Same text, use only once
                combined_docs[param] = inline
            else:
                # Different text: concatenate with newline separator
                combined_docs[param] = f"{inline}\n{docstring}"
        elif inline:
            # Only inline comment
            combined_docs[param] = inline
        elif docstring:
            # Only docstring Args
            combined_docs[param] = docstring

    return combined_docs


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


def _extract_args_from_docstring(docstring: str) -> Dict[str, str]:
    """Extract parameter documentation from Args section in docstring.

    Supports both Google-style and NumPy-style docstrings.

    Args:
        docstring: The full docstring text

    Returns:
        Dict mapping parameter names to their documentation strings
    """
    import re

    if not docstring:
        return {}

    docs = {}

    # Find the Args section
    # Match "Args:" at start of line with optional whitespace
    args_match = re.search(r'\n\s*Args:\s*\n', docstring)
    if not args_match:
        return {}

    # Get text starting from Args section
    args_start = args_match.end()

    # Find where Args section ends (next section or end of docstring)
    section_pattern = r'\n\s*(Returns|Raises|Yields|Examples?|Attributes?|Note|Notes|Warning|Warnings|See Also):'
    next_section = re.search(section_pattern, docstring[args_start:])

    if next_section:
        args_text = docstring[args_start:args_start + next_section.start()]
    else:
        args_text = docstring[args_start:]

    # Parse parameter entries
    # Pattern: param_name: description (Google style)
    # or param_name : description (NumPy style allows spaces)
    param_pattern = re.compile(r'^\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+?)(?=^\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:|$)',
                               re.MULTILINE | re.DOTALL)

    for match in param_pattern.finditer(args_text):
        param_name = match.group(1)
        param_doc = match.group(2).strip()

        # Clean up multi-line descriptions (remove excessive whitespace)
        param_doc = ' '.join(param_doc.split())

        docs[param_name] = param_doc

    return docs


def _generate_help_for_function(wrapper: ProtoWrapper) -> str:
    """Generate help text for a proto function."""
    lines = []

    # Get script name from prog override, sys.argv[0], or function name
    import sys
    from pathlib import Path

    if wrapper._prog:
        # Use explicitly set program name
        script_name = wrapper._prog
    elif sys.argv and sys.argv[0]:
        # Use argv[0] basename (works for .py files, executables, etc.)
        script_name = Path(sys.argv[0]).name
    else:
        # Fallback to function name
        script_name = wrapper.__name__

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

    # Description - only include text before Args:/Returns:/etc sections
    if wrapper.__doc__:
        doc = _extract_description_from_docstring(wrapper.__doc__)
        if doc:
            lines.append(doc)
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

        # Build description with default
        # Handle multi-line help text (from concatenated inline + docstring)
        help_lines = help_text.split('\n') if help_text else []

        # Build first line of description
        desc_parts = []
        if help_lines:
            desc_parts.append(help_lines[0])
        if default is not None:
            # Add default to the last line
            if len(help_lines) > 1:
                # Multi-line: add default to last line
                pass  # Will add below
            else:
                # Single line: add default to first line
                desc_parts.append(f"(default: {default})")

        first_line_desc = " ".join(desc_parts)

        # If the option string is too long, put description on next line
        if len(option_str) >= main_padding:
            lines.append(option_str)
            if first_line_desc:
                lines.append(" " * main_padding + first_line_desc)
            # Add subsequent lines from multi-line help
            for i, help_line in enumerate(help_lines[1:], start=1):
                if i == len(help_lines) - 1 and default is not None:
                    # Last line: add default
                    lines.append(" " * main_padding + help_line + f" (default: {default})")
                else:
                    lines.append(" " * main_padding + help_line)
        else:
            # Pad to align descriptions
            option_str = option_str.ljust(main_padding)
            lines.append(option_str + first_line_desc)
            # Add subsequent lines from multi-line help
            for i, help_line in enumerate(help_lines[1:], start=1):
                if i == len(help_lines) - 1 and default is not None:
                    # Last line: add default
                    lines.append(" " * main_padding + help_line + f" (default: {default})")
                else:
                    lines.append(" " * main_padding + help_line)

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


def _extract_description_from_docstring(docstring: str) -> str:
    """Extract the description part from a docstring (everything before Args:/Returns:/etc).

    Args:
        docstring: The full docstring text

    Returns:
        The description text with Args/Returns/etc sections removed
    """
    if not docstring:
        return ""

    doc = docstring.strip()

    # Look for common docstring sections (with optional leading whitespace)
    import re
    # Match section headers like "Args:", "Returns:", etc. at start of line with optional whitespace
    section_pattern = r'\n\s*(Args|Returns|Raises|Yields|Examples?|Attributes?|Note|Notes|Warning|Warnings|See Also):'

    match = re.search(section_pattern, doc)
    if match:
        # Return everything before the first section
        return doc[:match.start()].strip()

    return doc


def _convert_type(value: Any, annotation: Any) -> Any:
    """Convert a value to match the given type annotation."""
    # If value is already the right type or None, return as-is
    if value is None:
        return None

    # Get the origin type for generics like List[int]
    origin = get_origin(annotation)

    # Handle basic types
    if annotation == int or annotation is int:
        return int(value)
    elif annotation == float or annotation is float:
        return float(value)
    elif annotation == bool or annotation is bool:
        # Handle common boolean string representations
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)
    elif annotation == str or annotation is str:
        return str(value)

    # For complex types, try to return the value as-is
    return value


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


def proto(cls_or_func: Callable = None, *, cli: bool = False, prefix: bool = False, prog: str = None):
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
