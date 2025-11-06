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

  # Collect required parameters for positional args
  required_params = []
  for param_name, param_info in wrapper._params.items():
    if param_info.get("required", False) and param_info["annotation"] != bool:
      required_params.append(param_name)

  # Add positional arguments for required parameters
  for param_name in required_params:
    param_info = wrapper._params[param_name]
    annotation = param_info["annotation"]
    parser.add_argument(
      param_name,
      type=_get_converter_for_type(annotation),
      nargs='?',  # Make it optional so --flag syntax also works
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
        dest=f"_opt_{param_name}",
        default=None,
      )
      parser.add_argument(
        f"--no-{param_name.replace('_', '-')}",
        action="store_false",
        dest=f"_opt_{param_name}",
      )
    else:
      # Regular arguments (use different dest to avoid conflict with positional)
      parser.add_argument(
        arg_name,
        type=_get_converter_for_type(annotation),
        default=None,
        dest=f"_opt_{param_name}",
      )

  # Parse arguments
  args = parser.parse_args()

  # Build result by merging positional and optional arguments
  result = {}

  for param_name, param_info in wrapper._params.items():
    annotation = param_info["annotation"]
    default = param_info.get("default")
    required = param_info.get("required", False)

    # Check positional value (if this was a required param)
    pos_value = getattr(args, param_name, None) if param_name in required_params else None

    # Check optional flag value
    opt_value = getattr(args, f"_opt_{param_name}", None)

    # Determine final value (optional flags take precedence)
    if opt_value is not None:
      result[param_name] = opt_value
    elif pos_value is not None:
      result[param_name] = pos_value
    elif not required:
      # Use default for optional parameters
      result[param_name] = default
    else:
      # Required parameter with no value provided
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
