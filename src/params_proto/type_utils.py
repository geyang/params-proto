"""
Type utilities for params-proto.

Provides type conversion and type name extraction for CLI help generation.
"""

import ast
import inspect
from enum import Enum
from typing import Any, Union, get_args, get_origin, List, Tuple, Literal


def _convert_type(value: Any, annotation: Any) -> Any:
  """Convert a value to match the given type annotation.

  Args:
      value: The value to convert (can be a single value or list of values)
      annotation: The target type annotation

  Returns:
      Converted value matching the annotation type

  Security note: For dict types, we use ast.literal_eval which is safe for untrusted
  input - it only evaluates Python literal structures (strings, numbers, lists, dicts,
  etc.) and will never execute arbitrary code.
  """
  # If value is already the right type or None, return as-is
  if value is None:
    return None

  # Get the origin type for generics like List[int]
  origin = get_origin(annotation)

  # Handle Literal[...] types - validate against allowed values
  if origin is Literal:
    allowed_values = get_args(annotation)
    # For string values, check directly
    if isinstance(value, str):
      # Try to match string directly first
      if value in allowed_values:
        return value
      # Try to convert to matching type if value is string but allowed_values has other types
      for allowed in allowed_values:
        if isinstance(allowed, (int, float)):
          try:
            converted = type(allowed)(value)
            if converted in allowed_values:
              return converted
          except (ValueError, TypeError):
            pass
    # Direct check
    if value in allowed_values:
      return value
    raise ValueError(
      f"value must be one of {allowed_values}, got {repr(value)}"
    )

  # Handle Enum types - convert to enum member
  if inspect.isclass(annotation) and issubclass(annotation, Enum):
    # Try exact match first (case-sensitive)
    for member in annotation:
      if member.name == value:
        return member
    # Then try case-insensitive match for CLI convenience
    value_upper = str(value).upper()
    for member in annotation:
      if member.name.upper() == value_upper:
        return member
    # No match found
    valid_names = ", ".join(m.name for m in annotation)
    raise ValueError(
      f"'{value}' is not a valid {annotation.__name__}. "
      f"Valid options: {valid_names}"
    )

  # Handle dict types - parse safely using ast.literal_eval
  if origin is dict:
    try:
      parsed = ast.literal_eval(str(value))
      if not isinstance(parsed, dict):
        raise ValueError(f"expected dict, got {type(parsed).__name__}")
      return parsed
    except (ValueError, SyntaxError) as e:
      raise ValueError(f"invalid dict syntax: {e}")

  # Handle List[T] types
  if origin is list:
    args = get_args(annotation)
    element_type = args[0] if args else str

    # If value is already a list, convert each element
    if isinstance(value, list):
      return [_convert_type(item, element_type) for item in value]
    # If value is a single string, wrap it and convert
    else:
      return [_convert_type(value, element_type)]

  # Handle Tuple[T, ...] and Tuple[T1, T2, ...] types
  if origin is tuple:
    args = get_args(annotation)

    if not args:
      # Plain Tuple with no type args
      if isinstance(value, list):
        return tuple(value)
      else:
        return (value,)

    # Check if this is Tuple[T, ...] (variable-length) or Tuple[T1, T2, ...] (fixed-size)
    if len(args) == 2 and args[1] is Ellipsis:
      # Variable-length tuple: Tuple[int, ...]
      element_type = args[0]
      if isinstance(value, list):
        return tuple(_convert_type(item, element_type) for item in value)
      else:
        return (_convert_type(value, element_type),)
    else:
      # Fixed-size tuple: Tuple[int, str, float]
      if isinstance(value, list):
        converted = []
        for i, item in enumerate(value):
          # Use corresponding type annotation if available, otherwise treat as string
          item_type = args[i] if i < len(args) else str
          converted.append(_convert_type(item, item_type))
        return tuple(converted)
      else:
        # Single value for fixed-size tuple
        return (_convert_type(value, args[0]),) if args else (value,)

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

  # General fallback: try to instantiate custom classes (Path, dataclasses, etc.)
  # This handles any callable type annotation
  if callable(annotation) and not isinstance(value, annotation):
    try:
      return annotation(value)
    except (TypeError, ValueError):
      # If instantiation fails, return value as-is
      return value

  # For types we don't understand, return the value as-is
  return value


def _get_type_name(annotation: Any) -> str:
  """Get a human-readable type name for CLI help text.

  Args:
      annotation: The type annotation

  Returns:
      String representation like "INT", "FLOAT", "STR", or "LIST[INT]"
  """
  if annotation == int or annotation is int:
    return "INT"
  elif annotation == float or annotation is float:
    return "FLOAT"
  elif annotation == str or annotation is str:
    return "STR"
  elif annotation == bool or annotation is bool:
    return "BOOL"
  elif inspect.isclass(annotation) and issubclass(annotation, Enum):
    return f"{{{','.join(e.name for e in annotation)}}}"
  else:
    # Check if this is a generic type like List[T]
    origin = get_origin(annotation)

    if origin is Literal:
      args = get_args(annotation)
      return f"{{{','.join(repr(arg) for arg in args)}}}"
    elif origin is list:
      args = get_args(annotation)
      element_type_name = _get_type_name(args[0]) if args else "VALUE"
      return f"[{element_type_name}]"
    elif origin is tuple:
      args = get_args(annotation)
      if not args:
        return "(VALUE,)"
      elif len(args) == 2 and args[1] is Ellipsis:
        # Variable-length tuple: Tuple[int, ...]
        element_type_name = _get_type_name(args[0])
        return f"({element_type_name},...)"
      else:
        # Fixed-size tuple: Tuple[int, str, float]
        type_names = [_get_type_name(arg) for arg in args]
        return f"({','.join(type_names)})"
    elif origin is dict:
      args = get_args(annotation)
      if args and len(args) == 2:
        key_type = _get_type_name(args[0])
        val_type = _get_type_name(args[1])
        return f"{{{key_type}:{val_type}}}"
      return "{KEY:VALUE}"
    elif origin is Union:
      args = get_args(annotation)
      non_none_types = [arg for arg in args if arg is not type(None)]
      if len(non_none_types) == 1:
        # This is Optional[T], recursively get the type name of T
        return _get_type_name(non_none_types[0])

    # For custom classes like Path, dataclasses, etc.
    if inspect.isclass(annotation):
      return annotation.__name__.upper()

    return "VALUE"
