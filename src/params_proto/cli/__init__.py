"""
CLI argument parsing and help generation for params-proto.

This module handles:
- Command-line argument parsing
- Help text generation
- ANSI colorization for terminal output
"""

from params_proto.cli.cli_parse import parse_cli_args
from params_proto.cli.help_gen import _generate_help_for_function, _generate_help_for_subcommand
from params_proto.cli.ansi_help import colorize_help, get_terminal_width

__all__ = [
    "parse_cli_args",
    "colorize_help",
    "get_terminal_width",
]
