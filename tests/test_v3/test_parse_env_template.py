import os

from params_proto.parse_env_template import all_available, parse_env_template


def test_empty_template():
  result = parse_env_template("")
  assert result == [], "Expected an empty list for an empty template"


def test_no_variables():
  template = "This is a plain string with no variables."
  result = parse_env_template(template)
  assert result == [], "Expected an empty list for a template without variables"


def test_single_variable():
  template = "This string has a variable: $VAR_NAME/something-else."
  result = parse_env_template(template)
  assert result == ["VAR_NAME"], "Expected to extract the variable 'VAR_NAME'"


def test_multiple_variables():
  template = "String with $VAR1/something and blah$VAR2-other."
  result = parse_env_template(template)
  assert result == ["VAR1", "VAR2"], "Expected a list with all variable names"


def test_variable_with_numbers():
  template = "Testing variable with numbers: $VAR123/test."
  result = parse_env_template(template)
  assert result == ["VAR123"], "Expected to extract 'VAR123' as the variable name"


def test_underscores_in_variable_name():
  template = "Variable name with underscores: $VAR_WITH_UNDERSCORES/other."
  result = parse_env_template(template)
  assert result == ["VAR_WITH_UNDERSCORES"], (
    "Expected to extract variable name with underscores"
  )


def test_mixed_template():
  template = "Mixed template with variables $VAR1/something, text, and blah$VAR2."
  result = parse_env_template(template)
  assert result == ["VAR1", "VAR2"], (
    "Expected to extract all variable names while ignoring plain text"
  )


def test_all_available_with_no_variables():
  template = "This is a plain string with no variables."
  result = all_available(template)
  assert result is True, "Expected True for a template with no variables"


def test_all_available_with_all_variables_present():
  os.environ["VAR1"] = "value1"
  os.environ["VAR2"] = "value2"
  template = "Template with $VAR1 and $VAR2."
  result = all_available(template)
  assert result is True, (
    "Expected True when all variables are present in the environment"
  )
  # Clean up
  del os.environ["VAR1"]
  del os.environ["VAR2"]


def test_all_available_with_missing_variable():
  os.environ["VAR1"] = "value1"
  template = "Template with $VAR1 and $VAR2."
  result = all_available(template)
  assert result is False, (
    "Expected False when a variable is missing from the environment"
  )
  # Clean up
  del os.environ["VAR1"]


def test_all_available_with_empty_variable_strict_true():
  os.environ["VAR1"] = ""
  template = "Template with $VAR1."
  result = all_available(template, strict=True)
  assert result is False, "Expected False when a variable is empty and strict=True"
  # Clean up
  del os.environ["VAR1"]


def test_all_available_with_empty_variable_strict_false():
  os.environ["VAR1"] = ""
  template = "Template with $VAR1."
  result = all_available(template, strict=False)
  assert result is True, "Expected True when a variable is empty and strict=False"
  # Clean up
  del os.environ["VAR1"]
