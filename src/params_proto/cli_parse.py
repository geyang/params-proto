"""
CLI argument parsing for params-proto v3.

Converts sys.argv into kwargs for proto-decorated functions.
"""

import argparse
import sys
from typing import Any, Dict

from params_proto.type_utils import _convert_type


def parse_cli_args(wrapper) -> Dict[str, Any]:
  """Parse CLI arguments for a ProtoWrapper.

  Args:
      wrapper: ProtoWrapper instance

  Returns:
      Dictionary of parsed arguments
  """
  # Create parser (suppress help since we handle it ourselves)
  parser = argparse.ArgumentParser(
    description=wrapper.__doc__,
    add_help=False,  # We handle --help manually
  )

  # Add parameters
  # First pass: add required parameters as positional arguments
  # Second pass: add all parameters as optional --flag arguments
  for param_name, param_info in wrapper._params.items():
    annotation = param_info["annotation"]
    default = param_info.get("default")
    required = param_info.get("required", False)

    # Add positional argument for required parameters (non-boolean)
    if required and annotation != bool:
      parser.add_argument(
        param_name,
        type=_get_converter_for_type(annotation),
        nargs='?',  # Make it optional so --flag syntax also works
        dest=f"_pos_{param_name}",  # Use temporary dest to avoid conflict
      )

  # Add all parameters as optional --flag arguments
  for param_name, param_info in wrapper._params.items():
    arg_name = f"--{param_name.replace('_', '-')}"
    annotation = param_info["annotation"]
    default = param_info.get("default")
    required = param_info.get("required", False)

    # Handle boolean flags specially
    if annotation == bool:
      # Boolean flags use --flag / --no-flag pattern
      parser.add_argument(
        arg_name,
        action="store_true",
        dest=param_name,
        default=default,
      )
      parser.add_argument(
        f"--no-{param_name.replace('_', '-')}",
        action="store_false",
        dest=param_name,
      )
    else:
      # Regular arguments
      parser.add_argument(
        arg_name,
        type=_get_converter_for_type(annotation),
        default=default if not required else argparse.SUPPRESS,
        required=False,  # Never required since we have positional option
        dest=param_name,
      )

  # Parse arguments
  args = parser.parse_args()

  # Convert to dict and merge positional args with named args
  result = vars(args)

  # Merge positional arguments (prefer positional if both exist)
  for param_name in wrapper._params.keys():
    pos_key = f"_pos_{param_name}"
    if pos_key in result:
      pos_value = result.pop(pos_key)
      # If positional value was provided, use it (overrides named arg)
      if pos_value is not None:
        result[param_name] = pos_value
      # If positional wasn't provided but named arg wasn't either,
      # and it's required, we need to raise an error
      elif param_name not in result and wrapper._params[param_name].get("required", False):
        raise SystemExit(f"error: the following argument is required: {param_name}")

  return result


def _get_converter_for_type(annotation):
  """Get a converter function for the given type annotation.

  Args:
      annotation: Type annotation

  Returns:
      Converter function for argparse (just the type itself)
  """
  # Use the type itself as the constructor
  # Works for int, float, str, and most types
  return annotation
