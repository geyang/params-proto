"""
ANSI-formatted help text generation for terminal display.

Provides colorized, terminal-width-aware formatting of help text.
Keeps __help_str__ as plain text for testing, adds __ansi_str__ for display.

Terminal Detection Notes:
------------------------

Width Detection:
  - Uses shutil.get_terminal_size() which queries terminal dimensions
  - Checks COLUMNS env var first, then terminal ioctl
  - Width can change during a session (user resizes window)
  - For params-proto: No caching needed since help is printed once and exits
  - For long-running CLIs: Would need SIGWINCH handler or query-per-print
  - Current approach (query on each colorize_help() call) is correct and cheap

Color Detection:
  - Currently NOT implemented - colors always applied
  - Should check:
    * sys.stdout.isatty() - False for pipes/redirects
    * NO_COLOR env var - https://no-color.org/
    * TERM env var - 'dumb' or missing means no color support
  - TODO: Add should_use_color() function

Performance:
  - shutil.get_terminal_size() is fast (just reads terminal attributes)
  - No need to cache width since help is only printed once per execution
  - For high-frequency printing, would consider caching with SIGWINCH handler
"""

import re
import shutil
import textwrap
from typing import Optional


# ANSI color codes
class Colors:
  """ANSI color codes for terminal formatting."""

  # Reset
  RESET = "\033[0m"

  # Text styles
  BOLD = "\033[1m"
  DIM = "\033[2m"

  # Foreground colors
  RED = "\033[31m"
  GREEN = "\033[32m"
  YELLOW = "\033[33m"
  BLUE = "\033[34m"
  MAGENTA = "\033[35m"
  CYAN = "\033[36m"
  WHITE = "\033[37m"

  # Bright foreground colors
  BRIGHT_BLACK = "\033[90m"
  BRIGHT_RED = "\033[91m"
  BRIGHT_GREEN = "\033[92m"
  BRIGHT_YELLOW = "\033[93m"
  BRIGHT_BLUE = "\033[94m"
  BRIGHT_MAGENTA = "\033[95m"
  BRIGHT_CYAN = "\033[96m"


def get_terminal_width(default: int = 80, max_width: int = 120) -> int:
  """Get the current terminal width.

  Args:
      default: Default width if terminal size cannot be detected
      max_width: Maximum width to use even if terminal is wider

  Returns:
      Terminal width in characters
  """
  try:
    size = shutil.get_terminal_size(fallback=(default, 24))
    width = size.columns
    # Cap at max_width for readability
    return min(width, max_width)
  except Exception:
    return default


def strip_ansi(text: str) -> str:
  """Remove ANSI escape codes from text.

  Args:
      text: Text potentially containing ANSI codes

  Returns:
      Plain text with ANSI codes removed
  """
  ansi_escape = re.compile(r'\033\[[0-9;]*m')
  return ansi_escape.sub('', text)


def wrap_text_with_ansi(text: str, width: int, indent: str = "") -> list:
  """Wrap text while preserving ANSI codes.

  Args:
      text: Text to wrap (may contain ANSI codes)
      width: Target width for wrapping
      indent: Indentation string for wrapped lines

  Returns:
      List of wrapped lines
  """
  # For simplicity, strip ANSI for length calculation, then wrap
  plain = strip_ansi(text)

  if not plain.strip():
    return []

  # Calculate visible width
  visible_width = width - len(indent)

  if len(plain) <= visible_width:
    return [text]

  # Wrap the plain text
  wrapper = textwrap.TextWrapper(
    width=visible_width,
    initial_indent="",
    subsequent_indent="",
    break_long_words=False,
    break_on_hyphens=False,
  )

  wrapped_lines = wrapper.wrap(plain)

  # For now, just add indent to each line
  # (Preserving ANSI codes during wrapping is complex,
  #  so we'll apply formatting after wrapping)
  return [indent + line if i > 0 else line for i, line in enumerate(wrapped_lines)]


def colorize_help(help_str: str, width: Optional[int] = None) -> str:
  """Add ANSI colors and line wrapping to help text.

  Colorization:
  - Type names (INT, STR, FLOAT, etc.) -> bold bright blue (\x1b[1m\x1b[94m)
  - (required) -> bold red (\x1b[1m\x1b[31m)
  - (default: value) -> cyan parentheses with bold cyan value
    Example: \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m128\x1b[0m\x1b[36m)\x1b[0m
  - Option names (--foo) -> plain text (no formatting)

  Args:
      help_str: Plain text help string
      width: Terminal width (auto-detected if None)

  Returns:
      ANSI-formatted help text with proper line wrapping
  """
  if width is None:
    width = get_terminal_width()

  lines = help_str.split('\n')
  result = []

  for line in lines:
    # Skip empty lines and usage/section headers
    if not line.strip() or line.startswith('usage:') or line.endswith('options:'):
      result.append(line)
      continue

    # Check if this is an option line (starts with spaces and --)
    if re.match(r'^\s+--', line):
      # Parse the option line: "  --option TYPE     description..."
      # Match: (indent)(--option-name)( TYPE)?( spaces)(description)
      # TYPE must be uppercase (INT, STR, FLOAT, VALUE) or enum choices ({A,B,C})
      # to avoid mistakenly matching first description word as type
      match = re.match(r'^(\s+)(--\S+)(\s+(?:[A-Z]+|\{[^}]+\}))?(\s+)(.*)$', line)

      if match:
        indent_str = match.group(1)  # Leading spaces
        option_name = match.group(2)  # --option-name
        type_part = match.group(3) or ""  # TYPE (with leading space)
        spacing = match.group(4)  # Spaces between type and description
        description = match.group(5)  # Description text

        # Build the option part (everything before description)
        option_part = indent_str + option_name + type_part

        # Colorize components (no bold for option names)
        colored_option = f'{indent_str}{option_name}'
        if type_part:
          type_name = type_part.strip()
          colored_type = f' {Colors.BOLD}{Colors.BRIGHT_BLUE}{type_name}{Colors.RESET}'
        else:
          colored_type = ''

        # Colorize description
        colored_desc = _colorize_description(description)

        # Calculate padding needed to align descriptions
        # The description should start at the same column as in original
        desc_start_col = len(option_part) + len(spacing)

        # If the line would be too long, put description on next line
        full_line = option_part + spacing + description
        if len(full_line) > width:
          # Description goes on next line
          result.append(colored_option + colored_type)

          # Wrap the description
          indent_spaces = ' ' * desc_start_col
          wrapped_desc = _wrap_description(colored_desc, width, indent_spaces)
          result.extend(wrapped_desc)
        else:
          # Fits on one line
          padding = ' ' * (desc_start_col - len(option_part))
          result.append(colored_option + colored_type + padding + colored_desc)
      else:
        # Fallback: just colorize the line as-is
        result.append(_colorize_option_line(line))

    # Check if this is a continuation line (starts with lots of spaces, no --)
    elif re.match(r'^\s{20,}', line) and not re.match(r'^\s+--', line):
      # This is a description continuation line
      indent_match = re.match(r'^(\s+)(.*)$', line)
      if indent_match:
        indent = indent_match.group(1)
        text = indent_match.group(2)

        colored_text = _colorize_description(text)

        # Wrap if needed
        if len(line) > width:
          wrapped = _wrap_description(colored_text, width, indent)
          result.extend(wrapped)
        else:
          result.append(indent + colored_text)
      else:
        result.append(_colorize_continuation_line(line))

    else:
      # Other lines (help message, etc.)
      result.append(line)

  return '\n'.join(result)


def _colorize_description(text: str) -> str:
  """Colorize description text (handles (required) and (default: ...))."""
  # Bold red for (required)
  text = re.sub(
    r'\(required\)',
    f'{Colors.BOLD}{Colors.RED}(required){Colors.RESET}',
    text
  )

  # Cyan for (default: with bold cyan value
  text = re.sub(
    r'\(default:\s*([^)]+)\)',
    f'{Colors.CYAN}(default:{Colors.RESET} {Colors.BOLD}{Colors.CYAN}\\1{Colors.RESET}{Colors.CYAN}){Colors.RESET}',
    text
  )

  return text


def _wrap_description(text: str, width: int, indent: str) -> list:
  """Wrap description text preserving ANSI codes.

  Args:
      text: Description text (may contain ANSI codes)
      width: Terminal width
      indent: Indentation for continuation lines

  Returns:
      List of lines with proper indentation
  """
  # Strip ANSI for length calculations
  plain = strip_ansi(text)

  # Wrap the plain text
  wrapper = textwrap.TextWrapper(
    width=width,
    initial_indent=indent,
    subsequent_indent=indent,
    break_long_words=False,
    break_on_hyphens=False,
  )

  wrapped_plain = wrapper.wrap(plain)

  if not wrapped_plain:
    return []

  # Now we need to reapply ANSI codes
  # For simplicity, if text has ANSI codes, we'll just apply them to each wrapped line
  # This is a simplified approach - full ANSI preservation during wrapping is complex

  if strip_ansi(text) == text:
    # No ANSI codes, just return wrapped lines
    return wrapped_plain

  # Has ANSI codes - reapply to wrapped lines
  # Find (required) or (default:...) and colorize them
  result = []
  for line in wrapped_plain:
    colorized = _colorize_description(line)
    result.append(colorized)

  return result


def _colorize_option_line(line: str) -> str:
  """Colorize an option definition line.

  Example: "  --batch-size INT     Training batch size (default: 32)"
  """
  # Option names are not bolded (just left as-is)

  # Bold bright blue for type names (INT, STR, FLOAT, etc.)
  line = re.sub(
    r'\b(INT|STR|FLOAT|BOOL|VALUE)\b',
    f'{Colors.BOLD}{Colors.BRIGHT_BLUE}\\1{Colors.RESET}',
    line
  )

  # Bold red for (required)
  line = re.sub(
    r'\(required\)',
    f'{Colors.BOLD}{Colors.RED}(required){Colors.RESET}',
    line
  )

  # Cyan for (default: with bold cyan value
  line = re.sub(
    r'\(default:\s*([^)]+)\)',
    f'{Colors.CYAN}(default:{Colors.RESET} {Colors.BOLD}{Colors.CYAN}\\1{Colors.RESET}{Colors.CYAN}){Colors.RESET}',
    line
  )

  return line


def _colorize_continuation_line(line: str) -> str:
  """Colorize a description continuation line."""
  # Bold red for (required) on continuation lines
  line = re.sub(
    r'\(required\)',
    f'{Colors.BOLD}{Colors.RED}(required){Colors.RESET}',
    line
  )

  # Cyan for (default: with bold cyan value on continuation lines
  line = re.sub(
    r'\(default:\s*([^)]+)\)',
    f'{Colors.CYAN}(default:{Colors.RESET} {Colors.BOLD}{Colors.CYAN}\\1{Colors.RESET}{Colors.CYAN}){Colors.RESET}',
    line
  )

  return line


def get_ansi_help(wrapper) -> str:
  """Get ANSI-formatted help string from a proto wrapper.

  Args:
      wrapper: ProtoWrapper instance with __help_str__

  Returns:
      ANSI-formatted help text
  """
  if not hasattr(wrapper, '__help_str__'):
    return ""

  return colorize_help(wrapper.__help_str__)
