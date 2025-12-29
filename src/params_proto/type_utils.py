"""
Type utilities for params-proto.

Provides type conversion and type name extraction for CLI help generation.
"""

import inspect
from enum import Enum
from typing import Any, Union, get_args, get_origin, List


def _convert_type(value: Any, annotation: Any) -> Any:
  """Convert a value to match the given type annotation.

  Args:
      value: The value to convert (can be a single value or list of values)
      annotation: The target type annotation

  Returns:
      Converted value matching the annotation type
  """
  # If value is already the right type or None, return as-is
  if value is None:
    return None

  # Get the origin type for generics like List[int]
  origin = get_origin(annotation)

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

    if origin is list:
      args = get_args(annotation)
      element_type_name = _get_type_name(args[0]) if args else "VALUE"
      return f"[{element_type_name}]"
    elif origin is Union:
      args = get_args(annotation)
      non_none_types = [arg for arg in args if arg is not type(None)]
      if len(non_none_types) == 1:
        # This is Optional[T], recursively get the type name of T
        return _get_type_name(non_none_types[0])

    return "VALUE"
