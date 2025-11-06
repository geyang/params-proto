"""
Type utilities for params-proto.

Provides type conversion and type name extraction for CLI help generation.
"""

import inspect
from enum import Enum
from typing import Any, get_origin


def _convert_type(value: Any, annotation: Any) -> Any:
  """Convert a value to match the given type annotation.

  Args:
      value: The value to convert
      annotation: The target type annotation

  Returns:
      Converted value matching the annotation type
  """
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
  """Get a human-readable type name for CLI help text.

  Args:
      annotation: The type annotation

  Returns:
      String representation like "INT", "FLOAT", "STR", or ""
  """
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
