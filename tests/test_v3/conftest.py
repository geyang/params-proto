"""Shared fixtures for v3 CLI tests."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def run_cli():
  """Fixture for running CLI scripts in shell."""

  def _run(script_content: str, args: list[str] = None, expect_error: bool = False):
    """
    Run a Python script with CLI arguments.

    Args:
        script_content: Python code to execute
        args: List of command-line arguments
        expect_error: If True, expect non-zero exit code

    Returns:
        dict with keys: stdout, stderr, returncode
    """
    args = args or []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
      f.write(script_content)
      script_path = f.name

    try:
      result = subprocess.run(
        [sys.executable, script_path] + args,
        capture_output=True,
        text=True,
        timeout=5,
      )

      if not expect_error and result.returncode != 0:
        pytest.fail(
          f"Script failed with exit code {result.returncode}\n"
          f"stdout: {result.stdout}\n"
          f"stderr: {result.stderr}"
        )

      return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
      }
    finally:
      Path(script_path).unlink(missing_ok=True)

  return _run
