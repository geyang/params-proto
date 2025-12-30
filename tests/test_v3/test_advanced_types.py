"""
Tests for advanced type support: Literal, Enum, Path, and dict.

Tests comprehensive support for:
- Literal[...] with validation
- Enum conversion
- Path instantiation (and general callable types)
- dict parsing with ast.literal_eval
"""

import sys
import tempfile
import subprocess
from textwrap import dedent
import pytest


@pytest.fixture
def run_cli():
  """Fixture to run CLI scripts with arguments."""

  def _run(script_content: str, args: list[str] = None, expect_error: bool = False):
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
      import os
      os.unlink(script_path)

  return _run


# ============================================================================
# Literal[...] Tests - Validation of allowed values
# ============================================================================


def test_literal_string_values(run_cli):
  """Test Literal with string values."""
  script = dedent("""
    from typing import Literal
    from params_proto import proto

    @proto.cli
    def train(optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"):
        print(f"optimizer={optimizer}")

    if __name__ == "__main__":
        train()
    """)

  # Test valid value
  result = run_cli(script, ["--optimizer", "sgd"], expect_error=False)
  assert "optimizer=sgd" in result["stdout"]

  # Test default value
  result = run_cli(script, [], expect_error=False)
  assert "optimizer=adam" in result["stdout"]

  # Test invalid value
  result = run_cli(script, ["--optimizer", "invalid"], expect_error=True)
  assert result["returncode"] != 0
  assert "invalid" in result["stderr"].lower() or "value must be" in result["stderr"].lower()


def test_literal_numeric_values(run_cli):
  """Test Literal with numeric values."""
  script = dedent("""
    from typing import Literal
    from params_proto import proto

    @proto.cli
    def config(precision: Literal[16, 32, 64] = 32):
        print(f"precision={precision}")

    if __name__ == "__main__":
        config()
    """)

  # Test valid numeric value
  result = run_cli(script, ["--precision", "16"], expect_error=False)
  assert "precision=16" in result["stdout"]

  # Test invalid numeric value
  result = run_cli(script, ["--precision", "8"], expect_error=True)
  assert result["returncode"] != 0


def test_literal_mixed_types(run_cli):
  """Test Literal with mixed string and numeric values."""
  script = dedent("""
    from typing import Literal
    from params_proto import proto

    @proto.cli
    def process(mode: Literal["auto", 16, 32, 64] = "auto"):
        print(f"mode={mode}")

    if __name__ == "__main__":
        process()
    """)

  # Test string value
  result = run_cli(script, ["--mode", "auto"], expect_error=False)
  assert "mode=auto" in result["stdout"]

  # Test numeric value
  result = run_cli(script, ["--mode", "32"], expect_error=False)
  assert "mode=32" in result["stdout"]


# ============================================================================
# Enum Tests - Conversion to enum members
# ============================================================================


def test_enum_auto_values(run_cli):
  """Test Enum with auto() values."""
  script = dedent("""
    from enum import Enum, auto
    from params_proto import proto

    class Optimizer(Enum):
        ADAM = auto()
        SGD = auto()
        RMSPROP = auto()

    @proto.cli
    def train(opt: Optimizer = Optimizer.ADAM):
        print(f"optimizer={opt.name},value={opt.value}")

    if __name__ == "__main__":
        train()
    """)

  # Test exact case
  result = run_cli(script, ["--opt", "SGD"], expect_error=False)
  assert "optimizer=SGD" in result["stdout"]
  assert "value=2" in result["stdout"]

  # Test lowercase (case-insensitive matching)
  result = run_cli(script, ["--opt", "adam"], expect_error=False)
  assert "optimizer=ADAM" in result["stdout"]


def test_enum_custom_values(run_cli):
  """Test Enum with custom string values."""
  script = dedent("""
    from enum import Enum
    from params_proto import proto

    class Device(Enum):
        CUDA = "cuda"
        CPU = "cpu"
        MPS = "mps"

    @proto.cli
    def run(device: Device = Device.CUDA):
        print(f"device={device.name},value={device.value}")

    if __name__ == "__main__":
        run()
    """)

  result = run_cli(script, ["--device", "CPU"], expect_error=False)
  assert "device=CPU" in result["stdout"]
  assert "value=cpu" in result["stdout"]

  # Test case-insensitive
  result = run_cli(script, ["--device", "mps"], expect_error=False)
  assert "device=MPS" in result["stdout"]


def test_enum_invalid_member(run_cli):
  """Test Enum with invalid member name."""
  script = dedent("""
    from enum import Enum, auto
    from params_proto import proto

    class Color(Enum):
        RED = auto()
        GREEN = auto()
        BLUE = auto()

    @proto.cli
    def paint(color: Color = Color.RED):
        print(f"color={color.name}")

    if __name__ == "__main__":
        paint()
    """)

  result = run_cli(script, ["--color", "YELLOW"], expect_error=True)
  assert result["returncode"] != 0


# ============================================================================
# Path Tests - Instantiation of pathlib.Path
# ============================================================================


def test_path_relative(run_cli):
  """Test Path with relative path."""
  script = dedent("""
    from pathlib import Path
    from params_proto import proto

    @proto.cli
    def process(input_dir: Path = Path(".")):
        print(f"path={input_dir},type={type(input_dir).__name__}")

    if __name__ == "__main__":
        process()
    """)

  result = run_cli(script, ["--input-dir", "./data"], expect_error=False)
  assert "path=data" in result["stdout"]
  assert "type=PosixPath" in result["stdout"] or "type=WindowsPath" in result["stdout"]


def test_path_absolute(run_cli):
  """Test Path with absolute path."""
  script = dedent("""
    from pathlib import Path
    from params_proto import proto

    @proto.cli
    def process(output: Path = Path("/tmp")):
        print(f"path={output},type={type(output).__name__}")

    if __name__ == "__main__":
        process()
    """)

  result = run_cli(script, ["--output", "/var/log"], expect_error=False)
  assert "path=/var/log" in result["stdout"]
  assert "type=PosixPath" in result["stdout"] or "type=WindowsPath" in result["stdout"]


# ============================================================================
# dict Tests - Parsing with ast.literal_eval (security-safe)
# ============================================================================


def test_dict_simple(run_cli):
  """Test simple dict parsing."""
  script = dedent("""
    from typing import Dict
    from params_proto import proto

    @proto.cli
    def config(params: Dict[str, int] = None):
        print(f"params={params}")

    if __name__ == "__main__":
        config()
    """)

  # Test with double quotes
  result = run_cli(script, ['--params', '{"lr": 10, "batch_size": 32}'], expect_error=False)
  assert "params={'lr': 10, 'batch_size': 32}" in result["stdout"]

  # Test with single quotes
  result = run_cli(script, ["--params", "{'a': 1, 'b': 2}"], expect_error=False)
  assert "params={'a': 1, 'b': 2}" in result["stdout"]


def test_dict_nested(run_cli):
  """Test nested dict parsing."""
  script = dedent("""
    from typing import Dict
    from params_proto import proto

    @proto.cli
    def config(settings: Dict = None):
        print(f"settings={settings}")

    if __name__ == "__main__":
        config()
    """)

  result = run_cli(
    script,
    ['--settings', '{"model": {"layers": 3, "hidden": 128}, "data": {"split": 0.8}}'],
    expect_error=False,
  )
  assert "model" in result["stdout"]
  assert "layers" in result["stdout"]


def test_dict_with_various_types(run_cli):
  """Test dict with mixed value types."""
  script = dedent("""
    from typing import Dict
    from params_proto import proto

    @proto.cli
    def config(settings: Dict = None):
        print(f"settings={settings}")

    if __name__ == "__main__":
        config()
    """)

  # Dict with strings, ints, floats, booleans, lists (Python syntax, not JSON)
  result = run_cli(
    script,
    ['--settings', '{"name": "exp1", "count": 42, "rate": 0.5, "debug": True, "tags": ["ml", "v2"]}'],
    expect_error=False,
  )
  assert "name" in result["stdout"]
  assert "exp1" in result["stdout"]
  assert "42" in result["stdout"]
  assert "debug" in result["stdout"]


def test_dict_invalid_syntax(run_cli):
  """Test dict with invalid syntax."""
  script = dedent("""
    from typing import Dict
    from params_proto import proto

    @proto.cli
    def config(params: Dict = None):
        print(f"params={params}")

    if __name__ == "__main__":
        config()
    """)

  # Invalid dict syntax should fail
  result = run_cli(script, ["--params", "{invalid: json}"], expect_error=True)
  assert result["returncode"] != 0


def test_dict_not_dict_error(run_cli):
  """Test that non-dict inputs are rejected."""
  script = dedent("""
    from typing import Dict
    from params_proto import proto

    @proto.cli
    def config(params: Dict = None):
        print(f"params={params}")

    if __name__ == "__main__":
        config()
    """)

  # List instead of dict should fail
  result = run_cli(script, ["--params", "[1, 2, 3]"], expect_error=True)
  assert result["returncode"] != 0


# ============================================================================
# Combined Tests - Multiple advanced types together
# ============================================================================


def test_combined_literal_enum_dict(run_cli):
  """Test using Literal, Enum, and dict in same function."""
  script = dedent("""
    from typing import Literal, Dict
    from enum import Enum, auto
    from params_proto import proto

    class Backend(Enum):
        TORCH = auto()
        JAX = auto()

    @proto.cli
    def train(
        optimizer: Literal["adam", "sgd"] = "adam",
        backend: Backend = Backend.TORCH,
        config: Dict = None,
    ):
        print(f"opt={optimizer},backend={backend.name},config={config}")

    if __name__ == "__main__":
        train()
    """)

  result = run_cli(
    script,
    [
      "--optimizer", "sgd",
      "--backend", "JAX",
      "--config", '{"lr": 0.01, "epochs": 10}',
    ],
    expect_error=False,
  )
  assert "opt=sgd" in result["stdout"]
  assert "backend=JAX" in result["stdout"]
  assert "lr" in result["stdout"]


# ============================================================================
# Help Text Tests
# ============================================================================


def test_literal_help_text(run_cli):
  """Test help text generation for Literal types."""
  script = dedent("""
    from typing import Literal
    from params_proto import proto

    @proto.cli
    def train(optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"):
        '''Train model.'''
        pass

    if __name__ == "__main__":
        train()
    """)

  result = run_cli(script, ["--help"], expect_error=False)
  assert "--optimizer" in result["stdout"]
  # Should show allowed values
  assert ("adam" in result["stdout"] or "sgd" in result["stdout"])


def test_enum_help_text(run_cli):
  """Test help text generation for Enum types."""
  script = dedent("""
    from enum import Enum, auto
    from params_proto import proto

    class Device(Enum):
        CUDA = auto()
        CPU = auto()

    @proto.cli
    def run(device: Device = Device.CUDA):
        '''Run on device.'''
        pass

    if __name__ == "__main__":
        run()
    """)

  result = run_cli(script, ["--help"], expect_error=False)
  assert "--device" in result["stdout"]
  assert "CUDA" in result["stdout"] or "CPU" in result["stdout"]


def test_dict_help_text(run_cli):
  """Test help text generation for dict types."""
  script = dedent("""
    from typing import Dict
    from params_proto import proto

    @proto.cli
    def config(params: Dict[str, int] = None):
        '''Configure parameters.'''
        pass

    if __name__ == "__main__":
        config()
    """)

  result = run_cli(script, ["--help"], expect_error=False)
  assert "--params" in result["stdout"]
