from typing import List
import re


def parse_env_template(template: str) -> List[str]:
    """
    Extract and return the environment variable names from a given string template.

    :param template: A string template that potentially contains environment variables
                     in the format $VAR_NAME, ${VAR_NAME}, or similar.
    :return: A list of environment variable names found in the template.
    """
    # Regex to match patterns like ${VAR_NAME}, $VAR_NAME
    # Matches: $VAR or ${VAR}
    pattern = r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?"
    return re.findall(pattern, template)


def all_available(template: str, strict=True) -> bool:
    """
    Check if all environment variables in the template are available in the current environment.

    :param template: A string template that potentially contains environment variables
                     in the format $VAR_NAME, ${VAR_NAME}, or similar.
    :param strict: If True, treat empty environment variables as undefined. Otherwise, treat them as defined.
    :return: True if all environment variables are available, False otherwise.
    """
    from os import environ

    for var in parse_env_template(template):
        if var not in environ or (strict and environ[var] == ""):
            return False

    return True
