"""
CLI argument parsing for params-proto v3.

Converts sys.argv into kwargs for proto-decorated functions.
Simple custom parser - no argparse dependency.
"""

import sys
from typing import Any, Dict, Union, get_args, get_origin, List

from params_proto.type_utils import _convert_type


def _is_union_type(annotation) -> bool:
  """Check if annotation is a Union type."""
  import types
  import typing

  # Handle typing.Union (e.g., Union[A, B])
  origin = get_origin(annotation)
  if origin is typing.Union:
    return True

  # Handle types.UnionType (Python 3.10+ syntax: A | B)
  if hasattr(types, "UnionType") and isinstance(annotation, types.UnionType):
    return True

  return False


def _get_union_classes(annotation) -> list:
  """Get list of classes from Union type annotation."""
  if not _is_union_type(annotation):
    return []

  # Get args works for both typing.Union and types.UnionType
  args = get_args(annotation)

  # For types.UnionType (A | B), if get_args() returns empty, use __args__
  if not args and hasattr(annotation, "__args__"):
    args = annotation.__args__

  return [arg for arg in args if isinstance(arg, type) and arg is not type(None)]


def _normalize_class_name(class_name: str) -> str:
  """Normalize class name to lowercase without separators.

  Removes all hyphens and underscores for comparison.
  This allows matching:
  - PerspectiveCamera → perspectivecamera
  - perspective-camera → perspectivecamera
  - perspectivecamera → perspectivecamera
  """
  return class_name.replace("-", "").replace("_", "").lower()


def _match_class_by_name(name: str, classes: list) -> Union[type, None]:
  """Match a string to one of the Union classes.

  Supports:
  - PascalCase (e.g., 'PerspectiveCamera')
  - lowercase (e.g., 'perspectivecamera')
  - kebab-case (e.g., 'perspective-camera')
  - Abbreviated first word (e.g., 'perspective' → PerspectiveCamera)
  """
  import re

  normalized = _normalize_class_name(name)

  for cls in classes:
    # Try exact match first
    if cls.__name__ == name:
      return cls

    # Try normalized match
    if _normalize_class_name(cls.__name__) == normalized:
      return cls

    # Try abbreviated match: 'perspective' should match 'PerspectiveCamera'
    # Extract first word from PascalCase class name
    words = re.findall(r"[A-Z][a-z]*", cls.__name__)
    if words and words[0].lower() == name.lower():
      return cls

  return None


def parse_cli_args(wrapper) -> Dict[str, Any]:
  """Parse CLI arguments for a ProtoWrapper.

  Args:
      wrapper: ProtoWrapper instance

  Returns:
      Dictionary of parsed arguments
  """
  # Import _SINGLETONS to handle prefix arguments
  from params_proto.proto import _SINGLETONS, ptype

  # Build lookup maps for fast parameter identification
  param_types = {}  # kebab-name -> (original_name, annotation, is_bool)
  required_params = []
  union_params = {}  # kebab-name -> (original_name, [union_classes])

  # Map function parameters
  for param_name, param_info in wrapper._params.items():
    kebab_name = param_name.replace("_", "-")
    annotation = param_info["annotation"]
    is_bool = annotation == bool

    # Check if this is a Union type
    if _is_union_type(annotation):
      union_classes = _get_union_classes(annotation)
      # Only treat as Union subcommand if there's more than one class
      # If there's exactly one class, it's Optional[T] and should be treated as a regular param
      if len(union_classes) > 1:
        union_params[kebab_name] = (param_name, union_classes)
        # Union parameters are handled specially
        if param_info.get("required", False):
          required_params.append(param_name)
        continue
      elif len(union_classes) == 1:
        # This is Optional[T], use the inner type for type conversion
        annotation = union_classes[0]

    # Check if this is a single class type (dataclass, etc.)
    # Treat as a "union" with one option to enable same syntax
    if isinstance(annotation, type) and annotation not in {
      int,
      str,
      float,
      bool,
      list,
      dict,
      tuple,
      set,
    }:
      union_params[kebab_name] = (param_name, [annotation])
      if param_info.get("required", False):
        required_params.append(param_name)
      continue

    param_types[kebab_name] = (param_name, annotation, is_bool)

    if param_info.get("required", False) and not is_bool:
      required_params.append(param_name)

  # Map prefix parameters
  prefix_params = {}  # "prefix.param-name" -> (singleton, param_name, annotation, is_bool)
  for singleton_name, singleton in _SINGLETONS.items():
    if isinstance(singleton, type) and isinstance(singleton, ptype):
      annotations = type.__getattribute__(singleton, "__proto_annotations__")

      for param_name, annotation in annotations.items():
        kebab_key = f"{singleton_name}.{param_name.replace('_', '-')}"
        is_bool = annotation == bool
        prefix_params[kebab_key] = (singleton, param_name, annotation, is_bool)

  # Parse arguments
  result = {}
  prefix_values = {}  # (singleton, param_name) -> value
  positional_values = []
  union_selections = {}  # param_name -> selected_class
  union_attrs = {}  # (param_name, attr_name) -> value

  args = sys.argv[1:]
  i = 0

  while i < len(args):
    arg = args[i]

    # Handle --no-flag for booleans
    if arg.startswith("--no-"):
      key = arg[5:]  # Remove '--no-'

      # Check if it's a function parameter
      if key in param_types:
        orig_name, annotation, is_bool = param_types[key]
        if is_bool:
          result[orig_name] = False
          i += 1
          continue

      # Check if it's a prefix parameter
      if key in prefix_params:
        singleton, param_name, annotation, is_bool = prefix_params[key]
        if is_bool:
          prefix_values[(singleton, param_name)] = False
          i += 1
          continue

      raise SystemExit(f"error: unrecognized argument: --no-{key}")

    # Handle --flag arguments
    elif arg.startswith("--"):
      key = arg[2:]  # Remove '--'

      # Check for Union selection syntax: --param:ClassName
      if ":" in key:
        param_part, class_part = key.split(":", 1)

        if param_part in union_params:
          orig_name, union_classes = union_params[param_part]
          selected_class = _match_class_by_name(class_part, union_classes)

          if selected_class is None:
            class_names = [cls.__name__ for cls in union_classes]
            raise SystemExit(
              f"error: invalid class '{class_part}' for --{param_part}, "
              f"expected one of: {', '.join(class_names)}"
            )

          union_selections[orig_name] = selected_class
          i += 1
          continue

        raise SystemExit(f"error: unrecognized argument: {arg}")

      # Check for Union attribute syntax: --param.attr value
      if "." in key:
        param_part, attr_part = key.split(".", 1)

        # Check if this is a Union parameter attribute
        if param_part in union_params:
          orig_name, union_classes = union_params[param_part]

          # Get the value
          if i + 1 >= len(args):
            raise SystemExit(f"error: argument --{key} requires a value")

          value_str = args[i + 1]
          # Store raw value, will convert when we know the class
          union_attrs[(orig_name, attr_part.replace("-", "_"))] = value_str
          i += 2
          continue

        # Fall through to check if it's a prefix parameter

      # Check if it's a function parameter
      if key in param_types:
        orig_name, annotation, is_bool = param_types[key]

        if is_bool:
          # Boolean flag without value
          result[orig_name] = True
          i += 1
        else:
          # Check if this is a List type
          origin = get_origin(annotation)
          is_list = origin is list

          if is_list:
            # Collect all following values until next flag
            values = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
              values.append(args[i])
              i += 1

            if not values:
              raise SystemExit(f"error: argument --{key} requires at least one value")

            try:
              value = _convert_type(values, annotation)
            except (ValueError, TypeError) as e:
              raise SystemExit(f"error: invalid value for --{key}: {e}")

            result[orig_name] = value
          else:
            # Get next argument as single value
            if i + 1 >= len(args):
              raise SystemExit(f"error: argument --{key} requires a value")

            value_str = args[i + 1]
            try:
              value = _convert_type(value_str, annotation)
            except (ValueError, TypeError):
              raise SystemExit(f"error: invalid value for --{key}: {value_str}")

            result[orig_name] = value
            i += 2
        continue

      # Check if it's a prefix parameter
      if key in prefix_params:
        singleton, param_name, annotation, is_bool = prefix_params[key]

        if is_bool:
          # Boolean flag without value
          prefix_values[(singleton, param_name)] = True
          i += 1
        else:
          # Check if this is a List type
          origin = get_origin(annotation)
          is_list = origin is list

          if is_list:
            # Collect all following values until next flag
            values = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
              values.append(args[i])
              i += 1

            if not values:
              raise SystemExit(f"error: argument --{key} requires at least one value")

            try:
              value = _convert_type(values, annotation)
            except (ValueError, TypeError) as e:
              raise SystemExit(f"error: invalid value for --{key}: {e}")

            prefix_values[(singleton, param_name)] = value
          else:
            # Get next argument as single value
            if i + 1 >= len(args):
              raise SystemExit(f"error: argument --{key} requires a value")

            value_str = args[i + 1]
            try:
              value = _convert_type(value_str, annotation)
            except (ValueError, TypeError):
              raise SystemExit(f"error: invalid value for --{key}: {value_str}")

            prefix_values[(singleton, param_name)] = value
            i += 2
        continue

      # Unknown argument
      raise SystemExit(f"error: unrecognized argument: {arg}")

    # Positional argument
    else:
      # Check if this could be a Union class name for a required Union parameter
      matched_union = False
      for param_kebab, (param_name, union_classes) in union_params.items():
        if param_name in required_params and param_name not in union_selections:
          # Try to match this positional arg as a class name
          selected_class = _match_class_by_name(arg, union_classes)
          if selected_class:
            union_selections[param_name] = selected_class
            matched_union = True
            i += 1
            break

      if not matched_union:
        positional_values.append(arg)
        i += 1

  # Assign positional arguments to required parameters
  for i, param_name in enumerate(required_params):
    if i < len(positional_values):
      param_info = wrapper._params[param_name]
      annotation = param_info["annotation"]

      try:
        value = _convert_type(positional_values[i], annotation)
      except (ValueError, TypeError):
        raise SystemExit(
          f"error: invalid value for {param_name}: {positional_values[i]}"
        )

      # Only set if not already set by --flag
      if param_name not in result:
        result[param_name] = value

  # Check for missing required parameters (but don't fill in defaults)
  # Defaults are handled by the caller to preserve bind context priority
  for param_name, param_info in wrapper._params.items():
    if param_name not in result and param_name not in union_selections:
      if param_info.get("required", False):
        raise SystemExit(f"error: the following argument is required: {param_name}")

  # Apply prefix values to singletons BEFORE instantiating Union classes
  # This ensures that singleton overrides are available when creating Union instances
  for (singleton, param_name), value in prefix_values.items():
    setattr(singleton, param_name, value)

  # Instantiate Union classes with collected attributes
  for param_name, selected_class in union_selections.items():
    # Collect attributes for this Union parameter
    attrs = {}
    for (union_param, attr_name), value_str in union_attrs.items():
      if union_param == param_name:
        # Get the type annotation for this attribute
        if hasattr(selected_class, "__annotations__"):
          attr_type = selected_class.__annotations__.get(attr_name, str)
          try:
            attrs[attr_name] = _convert_type(value_str, attr_type)
          except (ValueError, TypeError):
            raise SystemExit(
              f"error: invalid value for --{param_name.replace('_', '-')}.{attr_name.replace('_', '-')}: {value_str}"
            )
        else:
          # No annotations, treat as string
          attrs[attr_name] = value_str

    # If selected_class is a proto.prefix singleton, merge its overrides
    from params_proto.proto import _SINGLETONS, ptype

    for singleton_key, singleton in _SINGLETONS.items():
      if singleton is selected_class:
        # This is a proto.prefix class - merge overrides into attrs
        if isinstance(singleton, type) and isinstance(singleton, ptype):
          overrides = type.__getattribute__(singleton, "__proto_overrides__")
          # Overrides take precedence over defaults, but CLI attrs take precedence over overrides
          for key, value in overrides.items():
            if key not in attrs:  # Only use override if not explicitly set via CLI
              attrs[key] = value
        break

    # Instantiate the class with collected attributes
    try:
      instance = selected_class(**attrs)
      result[param_name] = instance
    except TypeError as e:
      raise SystemExit(f"error: failed to instantiate {selected_class.__name__}: {e}")

  return result
