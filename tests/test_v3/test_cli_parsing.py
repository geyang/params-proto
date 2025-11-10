"""
Integration tests for CLI functionality.

Tests actual command-line parsing with shell-like interface.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

import pytest


@pytest.fixture
def run_cli():
  """Fixture for running CLI scripts in shell."""

  def _run(script_content: str, args: list[str] = None, expect_error: bool = False):
    """
    Run a Python script with CLI arguments.

    Args:
        script_content: Python code to execute
        args: List of command-line arguments (e.g., ['--seed', '42'])
        expect_error: If True, expect non-zero exit code

    Returns:
        dict with keys: stdout, stderr, returncode
    """
    args = args or []

    # Create temporary script file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
      f.write(script_content)
      script_path = f.name

    try:
      # Run the script with arguments
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
      # Clean up temporary file
      Path(script_path).unlink(missing_ok=True)

  return _run


# Basic CLI Tests


def test_simple_arguments(run_cli):
  """Test parsing simple int, float, str arguments."""
  script = dedent("""
    from params_proto import proto

    @proto.cli
    def main(
        name: str = "default",
        count: int = 0,
        rate: float = 1.0,
    ):
        print(f"{name},{count},{rate}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["--name", "test", "--count", "42", "--rate", "3.14"])
  assert result["stdout"].strip() == "test,42,3.14"


def test_boolean_flags(run_cli):
  """Test boolean flag parsing."""
  script = dedent("""
    from params_proto import proto

    @proto.cli
    def main(
        verbose: bool = False,
        debug: bool = True,
    ):
        print(f"verbose={verbose},debug={debug}")

    if __name__ == "__main__":
        main()
    """)

  # Test --flag sets to True
  result = run_cli(script, ["--verbose"])
  assert result["stdout"].strip() == "verbose=True,debug=True"

  # Test --no-flag sets to False
  result = run_cli(script, ["--no-debug"])
  assert result["stdout"].strip() == "verbose=False,debug=False"


def test_required_arguments(run_cli):
  """Test required argument validation."""
  script = dedent("""
    from params_proto import proto

    @proto.cli
    def main(name: str, count: int = 0):
        print(f"{name},{count}")

    if __name__ == "__main__":
        main()
    """)

  # Should fail without required argument
  result = run_cli(script, [], expect_error=True)
  assert result["returncode"] != 0
  assert "required" in result["stderr"].lower()

  # Should succeed with required argument
  result = run_cli(script, ["--name", "test"])
  assert result["stdout"].strip() == "test,0"


def test_help_output(run_cli):
  """Test --help flag."""
  script = dedent("""
    from params_proto import proto

    @proto.cli
    def main(name: str = "test"):
        '''A test CLI.'''
        print(name)

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["--help"], expect_error=True)
  assert result["returncode"] == 0
  assert "usage:" in result["stdout"]
  assert "A test CLI" in result["stdout"]
  assert "--name" in result["stdout"]


# Prefix Argument Tests


def test_prefix_basic(run_cli):
  """Test basic prefix argument parsing."""
  script = dedent("""
    from params_proto import proto

    @proto.prefix
    class Model:
        name: str = "resnet50"
        size: int = 256

    @proto.cli
    def main(seed: int = 42):
        print(f"{Model.name},{Model.size},{seed}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(
    script, ["--model.name", "vit", "--model.size", "512", "--seed", "999"]
  )
  assert result["stdout"].strip() == "vit,512,999"


def test_prefix_boolean_flags(run_cli):
  """Test boolean flags in prefix classes."""
  script = dedent("""
    from params_proto import proto

    @proto.prefix
    class Config:
        enabled: bool = False
        debug: bool = True

    @proto.cli
    def main():
        print(f"enabled={Config.enabled},debug={Config.debug}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["--config.enabled", "--no-config.debug"])
  assert result["stdout"].strip() == "enabled=True,debug=False"


def test_multiple_prefixes(run_cli):
  """Test multiple @proto.prefix classes."""
  script = dedent("""
    from params_proto import proto

    @proto.prefix
    class Model:
        name: str = "resnet"

    @proto.prefix
    class Training:
        lr: float = 0.001

    @proto.cli
    def main():
        print(f"{Model.name},{Training.lr}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["--model.name", "vit", "--training.lr", "0.01"])
  assert result["stdout"].strip() == "vit,0.01"


# Error Handling Tests


def test_unknown_argument(run_cli):
  """Test error on unknown argument."""
  script = dedent("""
    from params_proto import proto

    @proto.cli
    def main(name: str = "test"):
        print(name)

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["--unknown"], expect_error=True)
  assert result["returncode"] != 0
  assert "unrecognized" in result["stderr"]


def test_invalid_type(run_cli):
  """Test error on invalid type conversion."""
  script = dedent("""
    from params_proto import proto

    @proto.cli
    def main(count: int = 0):
        print(count)

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["--count", "not-a-number"], expect_error=True)
  assert result["returncode"] != 0
  assert "invalid value" in result["stderr"]


def test_missing_value(run_cli):
  """Test error when argument value is missing."""
  script = dedent("""
    from params_proto import proto

    @proto.cli
    def main(name: str = "test"):
        print(name)

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["--name"], expect_error=True)
  assert result["returncode"] != 0
  assert "requires a value" in result["stderr"]


# Positional Argument Tests


def test_single_positional(run_cli):
  """Test single positional argument."""
  script = dedent("""
    from params_proto import proto

    @proto.cli
    def main(name: str, count: int = 0):
        print(f"{name},{count}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["myname"])
  assert result["stdout"].strip() == "myname,0"


def test_positional_and_named(run_cli):
  """Test mixing positional and named arguments."""
  script = dedent("""
    from params_proto import proto

    @proto.cli
    def main(name: str, count: int = 0):
        print(f"{name},{count}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["myname", "--count", "42"])
  assert result["stdout"].strip() == "myname,42"


def test_named_overrides_positional(run_cli):
  """Test that named argument overrides positional."""
  script = dedent("""
    from params_proto import proto

    @proto.cli
    def main(name: str):
        print(name)

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["--name", "override"])
  assert result["stdout"].strip() == "override"


# Union Subcommand Tests


def test_union_pascalcase_selection(run_cli):
  """Test Union class selection with PascalCase."""
  script = dedent("""
    from dataclasses import dataclass
    from params_proto import proto

    @dataclass
    class PerspectiveCamera:
        fov: float = 60.0

    @dataclass
    class OrthographicCamera:
        scale: float = 1.0

    @proto.cli
    def main(camera: PerspectiveCamera | OrthographicCamera):
        print(f"{camera.__class__.__name__},{camera.fov if hasattr(camera, 'fov') else camera.scale}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["--camera:PerspectiveCamera"])
  assert result["stdout"].strip() == "PerspectiveCamera,60.0"

  result = run_cli(script, ["--camera:OrthographicCamera"])
  assert result["stdout"].strip() == "OrthographicCamera,1.0"


def test_union_kebabcase_selection(run_cli):
  """Test Union class selection with kebab-case."""
  script = dedent("""
    from dataclasses import dataclass
    from params_proto import proto

    @dataclass
    class PerspectiveCamera:
        fov: float = 60.0

    @dataclass
    class OrthographicCamera:
        scale: float = 1.0

    @proto.cli
    def main(camera: PerspectiveCamera | OrthographicCamera):
        print(f"{camera.__class__.__name__}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["--camera:perspective-camera"])
  assert result["stdout"].strip() == "PerspectiveCamera"

  result = run_cli(script, ["--camera:orthographic-camera"])
  assert result["stdout"].strip() == "OrthographicCamera"


def test_union_lowercase_selection(run_cli):
  """Test Union class selection with all lowercase."""
  script = dedent("""
    from dataclasses import dataclass
    from params_proto import proto

    @dataclass
    class PerspectiveCamera:
        fov: float = 60.0

    @dataclass
    class OrthographicCamera:
        scale: float = 1.0

    @proto.cli
    def main(camera: PerspectiveCamera | OrthographicCamera):
        if isinstance(camera, PerspectiveCamera):
            print(f"{camera.__class__.__name__}:{camera.fov}")
        else:
            print(f"{camera.__class__.__name__}:{camera.scale}")

    if __name__ == "__main__":
        main()
    """)

  # Test basic selection
  result = run_cli(script, ["--camera:perspective-camera"])
  assert result["stdout"].strip() == "PerspectiveCamera:60.0"

  # Test with field override
  result = run_cli(script, ["--camera:perspective-camera", "--camera.fov", "90.0"])
  assert result["stdout"].strip() == "PerspectiveCamera:90.0"

  # Test OrthographicCamera selection with field override
  result = run_cli(script, ["--camera:orthographic-camera", "--camera.scale", "2.0"])
  assert result["stdout"].strip() == "OrthographicCamera:2.0"


def test_union_positional_selection(run_cli):
  """Test Union class selection as positional argument."""
  script = dedent("""
    from dataclasses import dataclass
    from params_proto import proto

    @dataclass
    class PerspectiveCamera:
        fov: float = 60.0

    @dataclass
    class OrthographicCamera:
        scale: float = 1.0

    @proto.cli
    def main(camera: PerspectiveCamera | OrthographicCamera):
        print(f"{camera.__class__.__name__}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["perspective-camera"])
  assert result["stdout"].strip() == "PerspectiveCamera"

  result = run_cli(script, ["orthographic-camera"])
  assert result["stdout"].strip() == "OrthographicCamera"


def test_union_attribute_setting(run_cli):
  """Test setting Union instance attributes."""
  script = dedent("""
    from dataclasses import dataclass
    from params_proto import proto

    @dataclass
    class PerspectiveCamera:
        fov: float = 60.0
        aspect: float = 1.33

    @dataclass
    class OrthographicCamera:
        scale: float = 1.0

    @proto.cli
    def main(camera: PerspectiveCamera | OrthographicCamera):
        if isinstance(camera, PerspectiveCamera):
            print(f"{camera.fov},{camera.aspect}")
        else:
            print(f"{camera.scale}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(
    script,
    ["--camera:PerspectiveCamera", "--camera.fov", "45", "--camera.aspect", "1.77"],
  )
  assert result["stdout"].strip() == "45.0,1.77"


def test_union_with_other_params(run_cli):
  """Test Union parameter mixed with regular parameters."""
  script = dedent("""
    from dataclasses import dataclass
    from params_proto import proto

    @dataclass
    class PerspectiveCamera:
        fov: float = 60.0

    @dataclass
    class OrthographicCamera:
        scale: float = 1.0

    @proto.cli
    def main(
        camera: PerspectiveCamera | OrthographicCamera,
        output: str = "render.png",
        verbose: bool = False
    ):
        print(f"{camera.__class__.__name__},{output},{verbose}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(
    script, ["--camera:PerspectiveCamera", "--output", "test.png", "--verbose"]
  )
  assert result["stdout"].strip() == "PerspectiveCamera,test.png,True"


def test_union_invalid_class(run_cli):
  """Test error on invalid Union class name."""
  script = dedent("""
    from dataclasses import dataclass
    from params_proto import proto

    @dataclass
    class PerspectiveCamera:
        fov: float = 60.0

    @dataclass
    class OrthographicCamera:
        scale: float = 1.0

    @proto.cli
    def main(camera: PerspectiveCamera | OrthographicCamera):
        print("ok")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["--camera:InvalidCamera"], expect_error=True)
  assert result["returncode"] != 0
  assert "invalid class" in result["stderr"]


def test_union_missing_required(run_cli):
  """Test error when required Union parameter is missing."""
  script = dedent("""
    from dataclasses import dataclass
    from params_proto import proto

    @dataclass
    class PerspectiveCamera:
        fov: float = 60.0

    @proto.cli
    def main(camera: PerspectiveCamera):
        print(camera)

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, [], expect_error=True)
  assert result["returncode"] != 0
  assert "required" in result["stderr"]


def test_union_typing_syntax(run_cli):
  """Test Union with typing.Union syntax (not just |)."""
  script = dedent("""
    from dataclasses import dataclass
    from typing import Union
    from params_proto import proto

    @dataclass
    class PerspectiveCamera:
        fov: float = 60.0

    @dataclass
    class OrthographicCamera:
        scale: float = 1.0

    @proto.cli
    def main(camera: Union[PerspectiveCamera, OrthographicCamera]):
        print(f"{camera.__class__.__name__}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["--camera:PerspectiveCamera"])
  assert result["stdout"].strip() == "PerspectiveCamera"


# Single Class Subcommand Tests


def test_single_class_pascalcase(run_cli):
  """Test single class selection with PascalCase."""
  script = dedent("""
    from dataclasses import dataclass
    from params_proto import proto

    @dataclass
    class PerspectiveCamera:
        fov: float = 60.0

    @proto.cli
    def main(camera: PerspectiveCamera):
        print(f"{camera.__class__.__name__},{camera.fov}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["--camera:PerspectiveCamera"])
  assert result["stdout"].strip() == "PerspectiveCamera,60.0"


def test_single_class_kebabcase(run_cli):
  """Test single class selection with kebab-case."""
  script = dedent("""
    from dataclasses import dataclass
    from params_proto import proto

    @dataclass
    class PerspectiveCamera:
        fov: float = 60.0

    @proto.cli
    def main(camera: PerspectiveCamera):
        print(f"{camera.__class__.__name__}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["--camera:perspective-camera"])
  assert result["stdout"].strip() == "PerspectiveCamera"


def test_single_class_attribute_setting(run_cli):
  """Test setting single class attributes."""
  script = dedent("""
    from dataclasses import dataclass
    from params_proto import proto

    @dataclass
    class PerspectiveCamera:
        fov: float = 60.0
        aspect: float = 1.33

    @proto.cli
    def main(camera: PerspectiveCamera):
        print(f"{camera.fov},{camera.aspect}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(
    script,
    ["--camera:PerspectiveCamera", "--camera.fov", "45", "--camera.aspect", "1.77"],
  )
  assert result["stdout"].strip() == "45.0,1.77"


def test_single_class_positional(run_cli):
  """Test single class as positional argument."""
  script = dedent("""
    from dataclasses import dataclass
    from params_proto import proto

    @dataclass
    class PerspectiveCamera:
        fov: float = 60.0

    @proto.cli
    def main(camera: PerspectiveCamera):
        print(f"{camera.__class__.__name__}")

    if __name__ == "__main__":
        main()
    """)

  result = run_cli(script, ["perspective-camera"])
  assert result["stdout"].strip() == "PerspectiveCamera"


def test_single_class_dot_syntax(run_cli):
  """Test that single class parameters support dot syntax for attribute overrides."""
  script = dedent("""
    from dataclasses import dataclass
    from params_proto import proto

    @dataclass
    class PerspectiveCamera:
        fov: float = 60.0
        aspect: float = 1.33
        near: float = 0.1

    @proto.cli
    def main(camera: PerspectiveCamera):
        print(f"{camera.__class__.__name__}:{camera.fov},{camera.aspect},{camera.near}")

    if __name__ == "__main__":
        main()
    """)

  # Test 1: Class selection with default values
  result = run_cli(script, ["perspective-camera"])
  assert result["stdout"].strip() == "PerspectiveCamera:60.0,1.33,0.1"

  # Test 2: Override single attribute using dot syntax
  result = run_cli(script, ["perspective-camera", "--camera.fov", "45"])
  assert result["stdout"].strip() == "PerspectiveCamera:45.0,1.33,0.1"

  # Test 3: Override multiple attributes using dot syntax
  result = run_cli(script, ["perspective-camera", "--camera.fov", "45", "--camera.aspect", "1.77", "--camera.near", "0.5"])
  assert result["stdout"].strip() == "PerspectiveCamera:45.0,1.77,0.5"

  # Test 4: Named class selection with dot syntax overrides
  result = run_cli(script, ["--camera:perspective-camera", "--camera.fov", "90"])
  assert result["stdout"].strip() == "PerspectiveCamera:90.0,1.33,0.1"


def test_prefix_override(run_cli):
  """Test custom prefix name for singleton registration and field overrides."""
  script = dedent("""
    from dataclasses import dataclass
    from params_proto import proto

    @proto.prefix("perspective")
    class PerspectiveCamera:
        fov: float = 60.0

    @proto.prefix
    class Config:
        name: str = "default"

    @proto.cli
    def main(camera: PerspectiveCamera):
        # Verify prefix was set correctly
        from params_proto.proto import _SINGLETONS
        assert "perspective" in _SINGLETONS
        assert "config" in _SINGLETONS
        print(f"{camera.__class__.__name__}:{camera.fov}")

    if __name__ == "__main__":
        main()
    """)

  # Test 1: Class selection with actual class name
  result = run_cli(script, ["--camera:perspective-camera"])
  assert result["stdout"].strip() == "PerspectiveCamera:60.0"

  # Test 2: Field override using custom prefix
  # When @proto.prefix("perspective") is used, CLI args should use the custom prefix
  result = run_cli(script, ["--camera:perspective-camera", "--perspective.fov", "80"])
  assert result["stdout"].strip() == "PerspectiveCamera:80.0"

  # Test 3: Field override using custom prefix with positional class selection
  result = run_cli(script, ["perspective-camera", "--perspective.fov", "90"])
  assert result["stdout"].strip() == "PerspectiveCamera:90.0"
