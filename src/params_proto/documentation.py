"""
Documentation extraction utilities for params-proto.

Extracts parameter documentation from inline comments and docstrings,
combining them intelligently for comprehensive help text.
"""

import inspect
import re
from typing import Any, Dict


def _extract_docs_from_source(obj: Any) -> Dict[str, str]:
  """Extract documentation from inline comments and docstrings.

  Combines inline comments with docstring Args section. If both exist,
  they are concatenated with newline separator (inline first, then docstring).

  Args:
      obj: Function or class to extract documentation from

  Returns:
      Dict mapping parameter names to their documentation strings
  """
  inline_docs = {}

  try:
    source = inspect.getsource(obj)
    lines = source.split("\n")

    # Pattern to match parameter definitions: identifier: type [= value]
    # Must have a type annotation after the colon (not just a colon followed by nothing or newline)
    param_pattern = re.compile(r"^\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*\w")

    # Docstring section headers to skip
    docstring_sections = {
      "Args",
      "Returns",
      "Raises",
      "Yields",
      "Examples",
      "Example",
      "Attributes",
      "Note",
      "Notes",
      "Warning",
      "Warnings",
      "See",
    }

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
  if hasattr(obj, "__doc__") and obj.__doc__:
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


def _extract_args_from_docstring(docstring: str) -> Dict[str, str]:
  """Extract parameter documentation from Args section in docstring.

  Supports both Google-style and NumPy-style docstrings.

  Args:
      docstring: The full docstring text

  Returns:
      Dict mapping parameter names to their documentation strings
  """
  if not docstring:
    return {}

  docs = {}

  # Find the Args section
  # Match "Args:" at start of line with optional whitespace
  args_match = re.search(r"\n\s*Args:\s*\n", docstring)
  if not args_match:
    return {}

  # Get text starting from Args section
  args_start = args_match.end()

  # Find where Args section ends (next section or end of docstring)
  section_pattern = r"\n\s*(Returns|Raises|Yields|Examples?|Attributes?|Note|Notes|Warning|Warnings|See Also):"
  next_section = re.search(section_pattern, docstring[args_start:])

  if next_section:
    args_text = docstring[args_start : args_start + next_section.start()]
  else:
    args_text = docstring[args_start:]

  # Parse parameter entries
  # Pattern: param_name: description (Google style)
  # or param_name : description (NumPy style allows spaces)
  # Note: Use \Z (end of string) not $ (end of line) to avoid stopping at newlines
  param_pattern = re.compile(
    r"^\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+?)(?=^\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:|\Z)",
    re.MULTILINE | re.DOTALL,
  )

  for match in param_pattern.finditer(args_text):
    param_name = match.group(1)
    param_doc = match.group(2).strip()

    # Clean up multi-line descriptions (remove excessive whitespace)
    param_doc = " ".join(param_doc.split())

    docs[param_name] = param_doc

  return docs


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

  # Match section headers like "Args:", "Returns:", etc. at start of line with optional whitespace
  section_pattern = r"\n\s*(Args|Returns|Raises|Yields|Examples?|Attributes?|Note|Notes|Warning|Warnings|See Also):"

  match = re.search(section_pattern, doc)
  if match:
    # Return everything before the first section
    return doc[: match.start()].strip()

  return doc


def _generate_param_description(param_name: str, func_name: str) -> str:
  """Auto-generate parameter description from name.

  Args:
      param_name: Name of the parameter
      func_name: Name of the function (used to extract verb)

  Returns:
      Auto-generated description string
  """
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
