"""
Help text generation for params-proto.

Generates comprehensive CLI help text from function signatures,
inline comments, and docstrings.
"""

import textwrap
from typing import TYPE_CHECKING

from params_proto.documentation import (
  _extract_description_from_docstring,
  _generate_param_description,
)
from params_proto.type_utils import _get_type_name

if TYPE_CHECKING:
  from params_proto.proto import ProtoWrapper


# Global singleton registry - imported from proto.py at runtime
# This is set by proto.py when it imports this module
_SINGLETONS = {}


def _wrap_text(text: str, width: int = 80, initial_indent: str = "", subsequent_indent: str = "") -> list:
  """Wrap text to specified width with proper indentation.

  Args:
      text: Text to wrap
      width: Maximum line width (default: 80)
      initial_indent: Indentation for first line
      subsequent_indent: Indentation for subsequent lines

  Returns:
      List of wrapped lines
  """
  wrapper = textwrap.TextWrapper(
    width=width,
    initial_indent=initial_indent,
    subsequent_indent=subsequent_indent,
    break_long_words=False,
    break_on_hyphens=False,
  )
  return wrapper.wrap(text)


def _generate_help_for_function(wrapper: "ProtoWrapper") -> str:
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
    kebab_name = name.replace('_', '-')
    type_name = _get_type_name(param_info["annotation"])
    # For booleans defaulting to True, show --no-flag (to disable)
    # For booleans defaulting to False, show --flag (to enable)
    if param_info["annotation"] == bool and wrapper._defaults.get(name) is True:
      arg_name = f"--no-{kebab_name}"
    else:
      arg_name = f"--{kebab_name}"
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
  help_padding = (
    "  -h, --help                   " if has_prefixed else "  -h, --help           "
  )

  lines.append(help_padding + "show this help message and exit")

  # Add parameters
  for name, param_info in wrapper._params.items():
    kebab_name = name.replace('_', '-')
    type_name = _get_type_name(param_info["annotation"])
    default = wrapper._defaults.get(name)
    is_required = param_info.get("required", False)
    help_text = wrapper._field_docs.get(name, "")
    # For booleans defaulting to True, show --no-flag (to disable)
    if param_info["annotation"] == bool and default is True:
      arg_name = f"--no-{kebab_name}"
    else:
      arg_name = f"--{kebab_name}"

    # Auto-generate description if missing
    if not help_text:
      help_text = _generate_param_description(name, wrapper.__name__)

    # Build the option line with proper spacing
    if type_name:
      option_str = f"  {arg_name} {type_name}"
    else:
      option_str = f"  {arg_name}"

    # Build description with default or required marker
    # Handle multi-line help text (from concatenated inline + docstring)
    help_lines = help_text.split("\n") if help_text else []

    # Build first line of description
    desc_parts = []
    if help_lines:
      desc_parts.append(help_lines[0])

    # Determine what to show: (required) or (default: value)
    # Check if help text already contains (required) or (default:...)
    help_text_lower = help_text.lower()
    has_required_in_help = "(required)" in help_text_lower
    has_default_in_help = "(default:" in help_text_lower

    if is_required and default is None and not has_required_in_help:
      # Required parameter with no default, and help doesn't already say (required)
      suffix = "(required)"
    elif default is not None and not has_default_in_help:
      # Optional parameter with default, and help doesn't already say (default:...)
      suffix = f"(default: {default})"
    else:
      # Has a default but it's None, or help already contains the marker
      suffix = None

    # Add suffix to appropriate line
    if len(help_lines) > 1:
      # Multi-line: add suffix to last line
      pass  # Will add below
    else:
      # Single line: add suffix to first line
      if suffix:
        desc_parts.append(suffix)

    first_line_desc = " ".join(desc_parts)

    # If the option string is too long, put description on next line
    if len(option_str) >= main_padding:
      lines.append(option_str)
      if first_line_desc:
        lines.append(" " * main_padding + first_line_desc)
      # Add subsequent lines from multi-line help
      for i, help_line in enumerate(help_lines[1:], start=1):
        if i == len(help_lines) - 1 and suffix:
          # Last line: add suffix (required or default)
          lines.append(" " * main_padding + help_line + f" {suffix}")
        else:
          lines.append(" " * main_padding + help_line)
    else:
      # Pad to align descriptions
      option_str = option_str.ljust(main_padding)
      lines.append(option_str + first_line_desc)
      # Add subsequent lines from multi-line help
      for i, help_line in enumerate(help_lines[1:], start=1):
        if i == len(help_lines) - 1 and suffix:
          # Last line: add suffix (required or default)
          lines.append(" " * main_padding + help_line + f" {suffix}")
        else:
          lines.append(" " * main_padding + help_line)

  # Add sections for any @proto.prefix singletons
  for singleton_name, singleton in _SINGLETONS.items():
    # Import ProtoWrapper and ptype here to avoid circular import
    from params_proto.proto import ProtoWrapper, ptype

    # Handle metaclass-based proto classes
    if isinstance(singleton, type) and isinstance(singleton, ptype):
      lines.append(f"\n{singleton_name.capitalize()} options:")
      if singleton.__doc__:
        lines.append(f"  {singleton.__doc__.strip()}")
        lines.append("")

      # Access proto metadata
      annotations = type.__getattribute__(singleton, "__proto_annotations__")
      defaults = type.__getattribute__(singleton, "__proto_defaults__")
      field_docs = type.__getattribute__(singleton, "__proto_field_docs__")

      for param_name, annotation in annotations.items():
        arg_name = f"--{singleton_name}.{param_name.replace('_', '-')}"
        type_name = _get_type_name(annotation)
        default = defaults.get(param_name)
        help_text = field_docs.get(param_name, "")

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


def _generate_help_for_subcommand(subcommand_class: type, script_name: str) -> str:
  """Generate help text for a Union subcommand class (e.g., PerspectiveCamera).

  Args:
      subcommand_class: The dataclass/class that represents the subcommand
      script_name: The script name to show in usage

  Returns:
      Help text string
  """
  from params_proto.proto import _pascal_to_kebab

  lines = []
  class_name = subcommand_class.__name__
  kebab_name = _pascal_to_kebab(class_name)

  # Usage line
  lines.append(f"\nusage: {script_name} {kebab_name} [-h] [OPTIONS]")
  lines.append("")

  # Description from docstring
  if subcommand_class.__doc__:
    lines.append(subcommand_class.__doc__.strip())
    lines.append("")

  # Options section
  lines.append("options:")
  lines.append("  -h, --help           show this help message and exit")

  # Get annotations and defaults
  annotations = getattr(subcommand_class, "__annotations__", {})

  # Get defaults from class attributes or dataclass fields
  defaults = {}
  if hasattr(subcommand_class, "__dataclass_fields__"):
    # Dataclass
    for field_name, field in subcommand_class.__dataclass_fields__.items():
      if field.default is not field.default_factory:
        defaults[field_name] = field.default
      elif field.default_factory is not field.default_factory:
        defaults[field_name] = field.default_factory()
  else:
    # Regular class - get defaults from class dict
    for name in annotations:
      if hasattr(subcommand_class, name):
        defaults[name] = getattr(subcommand_class, name)

  # Get field docs from source (inline comments)
  field_docs = {}
  try:
    import inspect
    source = inspect.getsource(subcommand_class)
    # Simple extraction of inline comments
    for line in source.split("\n"):
      if ":" in line and "#" in line:
        # Try to extract field name and comment
        parts = line.split("#", 1)
        if len(parts) == 2:
          field_part = parts[0].strip()
          comment = parts[1].strip()
          # Extract field name (before the colon)
          if ":" in field_part:
            field_name = field_part.split(":")[0].strip()
            if field_name in annotations:
              field_docs[field_name] = comment
  except (OSError, TypeError):
    pass

  # Add parameters
  for param_name, annotation in annotations.items():
    kebab_param = param_name.replace("_", "-")
    type_name = _get_type_name(annotation)
    default = defaults.get(param_name)
    help_text = field_docs.get(param_name, "")

    # Auto-generate description if missing
    if not help_text:
      help_text = _generate_param_description(param_name, class_name.lower())

    # Build the option line
    if type_name:
      option_str = f"  --{kebab_param} {type_name}"
    else:
      option_str = f"  --{kebab_param}"

    # Pad to align descriptions
    if len(option_str) < 23:
      option_str = option_str.ljust(23)
    else:
      option_str = option_str + "  "

    # Build description with default
    desc_parts = []
    if help_text:
      desc_parts.append(help_text)
    if default is not None:
      desc_parts.append(f"(default: {default})")

    lines.append(option_str + " ".join(desc_parts))

  lines.append("")
  return "\n".join(lines)
