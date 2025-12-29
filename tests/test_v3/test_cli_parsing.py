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


def test_optional_str_cli_parsing(run_cli):
  """Test that Optional[str] is parsed as simple optional parameter, not Union subcommand.

  ISSUE: Currently Optional[str] is incorrectly treated as Union[str, NoneType],
  requiring --param:str syntax instead of the normal --param value syntax.

  This test demonstrates the issue and documents the expected behavior.
  """
  script = dedent("""
    from typing import Optional
    from params_proto import proto

    @proto.cli
    def train(
        checkpoint: Optional[str] = None,  # Path to checkpoint file
        learning_rate: float = 0.001,
    ):
        print(f"checkpoint={checkpoint},lr={learning_rate}")

    if __name__ == "__main__":
        train()
    """)

  # Test 1: Optional[str] with value should work with --param value syntax
  result = run_cli(script, ["--checkpoint", "model.pt"], expect_error=False)
  assert result["stdout"].strip() == "checkpoint=model.pt,lr=0.001"

  # Test 2: Optional[str] without value should use None default
  result = run_cli(script, [], expect_error=False)
  assert result["stdout"].strip() == "checkpoint=None,lr=0.001"

  # Test 3: Optional[str] with other args
  result = run_cli(
    script,
    ["--checkpoint", "model.pt", "--learning-rate", "0.01"],
    expect_error=False,
  )
  assert result["stdout"].strip() == "checkpoint=model.pt,lr=0.01"


def test_optional_int_cli_parsing(run_cli):
  """Test that Optional[int] is parsed as simple optional parameter, not Union subcommand.

  Similar issue as Optional[str], this should work with normal --param value syntax.
  """
  script = dedent("""
    from typing import Optional
    from params_proto import proto

    @proto.cli
    def train(
        seed: Optional[int] = None,
        epochs: int = 10,
    ):
        print(f"seed={seed},epochs={epochs}")

    if __name__ == "__main__":
        train()
    """)

  # Test 1: Optional[int] with value
  result = run_cli(script, ["--seed", "42"], expect_error=False)
  assert result["stdout"].strip() == "seed=42,epochs=10"

  # Test 2: Optional[int] without value should use None default
  result = run_cli(script, [], expect_error=False)
  assert result["stdout"].strip() == "seed=None,epochs=10"

  # Test 3: Optional[int] with other args
  result = run_cli(
    script,
    ["--seed", "123", "--epochs", "50"],
    expect_error=False,
  )
  assert result["stdout"].strip() == "seed=123,epochs=50"


# List[T] Type CLI Parsing Tests


def test_list_str_cli_parsing(run_cli):
  """Test that List[str] can be parsed from CLI arguments.

  ISSUE: Currently List[str] is not properly handled in CLI parsing.
  The _convert_type function doesn't handle List types, so passing
  --tags value results in the string value not being wrapped in a list.
  Also, the CLI parser only consumes one value after --param, so multiple
  values are not collected into the list.

  This test documents the expected behavior when the bug is fixed.
  """
  script = dedent("""
    from typing import List
    from params_proto import proto

    @proto.cli
    def train(
        tags: List[str] = None,
        batch_size: int = 32,
    ):
        print(f"tags={tags},batch_size={batch_size}")

    if __name__ == "__main__":
        train()
    """)

  # Test 1: List[str] with single value should create list
  # CURRENTLY BROKEN: Returns "tags=experiment" instead of "tags=['experiment']"
  result = run_cli(script, ["--tags", "experiment"], expect_error=False)
  assert result["stdout"].strip() == "tags=['experiment'],batch_size=32"

  # Test 2: List[str] with multiple values should work
  # CURRENTLY BROKEN: Only first value is captured, rest become positional args
  # NOTE: This may require special CLI syntax like --tags tag1 tag2 or --tags tag1 --tags tag2
  result = run_cli(script, ["--tags", "exp1", "exp2"], expect_error=False)
  assert result["stdout"].strip() == "tags=['exp1', 'exp2'],batch_size=32"

  # Test 3: List[str] without value should use None/empty default
  result = run_cli(script, [], expect_error=False)
  assert result["stdout"].strip() == "tags=None,batch_size=32"

  # Test 4: List[str] with other args
  # CURRENTLY BROKEN: Only first value captured, "--batch-size" treated as positional
  result = run_cli(
    script,
    ["--tags", "tag1", "tag2", "--batch-size", "64"],
    expect_error=False,
  )
  assert result["stdout"].strip() == "tags=['tag1', 'tag2'],batch_size=64"


def test_list_int_cli_parsing(run_cli):
  """Test that List[int] can be parsed from CLI arguments.

  Similar to List[str], List[int] should support passing multiple integer
  values and converting them to a list. CURRENTLY BROKEN: Values are not
  converted to int and not wrapped in list.
  """
  script = dedent("""
    from typing import List
    from params_proto import proto

    @proto.cli
    def train(
        gpu_ids: List[int] = None,
        seed: int = 42,
    ):
        print(f"gpu_ids={gpu_ids},seed={seed}")

    if __name__ == "__main__":
        train()
    """)

  # Test 1: List[int] with single value
  # CURRENTLY BROKEN: Returns "gpu_ids=0" (as string) instead of "gpu_ids=[0]"
  result = run_cli(script, ["--gpu-ids", "0"], expect_error=False)
  assert result["stdout"].strip() == "gpu_ids=[0],seed=42"

  # Test 2: List[int] with multiple values
  # CURRENTLY BROKEN: Only first value captured, rest become positional args
  result = run_cli(script, ["--gpu-ids", "0", "1", "2"], expect_error=False)
  assert result["stdout"].strip() == "gpu_ids=[0, 1, 2],seed=42"

  # Test 3: List[int] without value should use None/empty default
  result = run_cli(script, [], expect_error=False)
  assert result["stdout"].strip() == "gpu_ids=None,seed=42"

  # Test 4: List[int] with other args
  # CURRENTLY BROKEN: Only first value captured, "--seed" might not parse correctly
  result = run_cli(
    script,
    ["--gpu-ids", "0", "1", "--seed", "99"],
    expect_error=False,
  )
  assert result["stdout"].strip() == "gpu_ids=[0, 1],seed=99"

  # Test 5: Invalid integer in List[int] should error
  # CURRENTLY BROKEN: Type conversion doesn't happen for List types
  result = run_cli(
    script,
    ["--gpu-ids", "invalid", "1"],
    expect_error=True,
  )
  assert result["returncode"] != 0
  assert "invalid value" in result["stderr"]


def test_list_float_cli_parsing(run_cli):
  """Test that List[float] can be parsed from CLI arguments.

  CURRENTLY BROKEN: Values are not converted to float and not wrapped in list.
  """
  script = dedent("""
    from typing import List
    from params_proto import proto

    @proto.cli
    def train(
        learning_rates: List[float] = None,
        momentum: float = 0.9,
    ):
        print(f"learning_rates={learning_rates},momentum={momentum}")

    if __name__ == "__main__":
        train()
    """)

  # Test 1: List[float] with single value
  # CURRENTLY BROKEN: Returns string instead of [float]
  result = run_cli(script, ["--learning-rates", "0.001"], expect_error=False)
  assert result["stdout"].strip() == "learning_rates=[0.001],momentum=0.9"

  # Test 2: List[float] with multiple values
  # CURRENTLY BROKEN: Only first value captured
  result = run_cli(
    script,
    ["--learning-rates", "0.001", "0.0001", "0.00001"],
    expect_error=False,
  )
  assert result["stdout"].strip() == "learning_rates=[0.001, 0.0001, 1e-05],momentum=0.9"

  # Test 3: List[float] without value should use None/empty default
  result = run_cli(script, [], expect_error=False)
  assert result["stdout"].strip() == "learning_rates=None,momentum=0.9"

  # Test 4: List[float] with other args
  # CURRENTLY BROKEN: Only first value captured, "--momentum" might not parse correctly
  result = run_cli(
    script,
    ["--learning-rates", "0.01", "0.001", "--momentum", "0.95"],
    expect_error=False,
  )
  assert result["stdout"].strip() == "learning_rates=[0.01, 0.001],momentum=0.95"


def test_list_with_defaults(run_cli):
  """Test List parameters with non-None default values.

  Test 1 works because no CLI parsing happens, but Tests 2-4 fail due to
  CLI parsing issues with List types.
  """
  script = dedent("""
    from typing import List
    from params_proto import proto

    @proto.cli
    def train(
        gpu_ids: List[int] = [0, 1],
        tags: List[str] = ["default"],
        learning_rates: List[float] = [0.001, 0.0001],
    ):
        print(f"gpu_ids={gpu_ids},tags={tags},learning_rates={learning_rates}")

    if __name__ == "__main__":
        train()
    """)

  # Test 1: Using default values - WORKS
  result = run_cli(script, [], expect_error=False)
  assert result["stdout"].strip() == "gpu_ids=[0, 1],tags=['default'],learning_rates=[0.001, 0.0001]"

  # Test 2: Overriding List[int] default
  # CURRENTLY BROKEN: Only first value captured as string
  result = run_cli(script, ["--gpu-ids", "2", "3", "4"], expect_error=False)
  assert result["stdout"].strip() == "gpu_ids=[2, 3, 4],tags=['default'],learning_rates=[0.001, 0.0001]"

  # Test 3: Overriding List[str] default
  # CURRENTLY BROKEN: Single value not wrapped in list
  result = run_cli(script, ["--tags", "custom"], expect_error=False)
  assert result["stdout"].strip() == "gpu_ids=[0, 1],tags=['custom'],learning_rates=[0.001, 0.0001]"

  # Test 4: Overriding all lists
  # CURRENTLY BROKEN: All values not properly converted and wrapped
  result = run_cli(
    script,
    [
      "--gpu-ids", "0",
      "--tags", "exp",
      "--learning-rates", "0.01",
    ],
    expect_error=False,
  )
  assert result["stdout"].strip() == "gpu_ids=[0],tags=['exp'],learning_rates=[0.01]"


def test_list_with_prefix_class(run_cli):
  """Test List parameters in @proto.prefix classes.

  CURRENTLY BROKEN: CLI parsing of List in prefix classes has same issues
  as regular parameters.
  """
  script = dedent("""
    from typing import List
    from params_proto import proto

    @proto.prefix
    class Model:
        layer_sizes: List[int] = [256, 128]
        activation_fns: List[str] = ["relu", "relu"]

    @proto.cli
    def train(seed: int = 42):
        print(f"layer_sizes={Model.layer_sizes},activations={Model.activation_fns},seed={seed}")

    if __name__ == "__main__":
        train()
    """)

  # Test 1: Using default values - WORKS
  result = run_cli(script, ["--seed", "99"], expect_error=False)
  assert result["stdout"].strip() == "layer_sizes=[256, 128],activations=['relu', 'relu'],seed=99"

  # Test 2: Overriding List in prefix class with multiple values
  result = run_cli(script, ["--model.layer-sizes", "512", "256", "128"], expect_error=False)
  assert result["stdout"].strip() == "layer_sizes=[512, 256, 128],activations=['relu', 'relu'],seed=42"

  # Test 3: Overriding both List parameters in prefix
  # CURRENTLY BROKEN: Multiple values not captured for List types
  result = run_cli(
    script,
    [
      "--model.layer-sizes", "1024", "512",
      "--model.activation-fns", "gelu", "gelu",
      "--seed", "123",
    ],
    expect_error=False,
  )
  assert result["stdout"].strip() == "layer_sizes=[1024, 512],activations=['gelu', 'gelu'],seed=123"


def test_list_empty_initialization(run_cli):
  """Test that List parameters can be initialized as empty lists.

  Test 1 works because no CLI parsing happens.
  """
  script = dedent("""
    from typing import List
    from params_proto import proto

    @proto.cli
    def main(
        ids: List[int] = [],
        names: List[str] = [],
    ):
        print(f"ids={ids},names={names}")

    if __name__ == "__main__":
        main()
    """)

  # Test 1: Using default empty lists - WORKS
  result = run_cli(script, [], expect_error=False)
  assert result["stdout"].strip() == "ids=[],names=[]"

  # Test 2: Override empty list with values
  # CURRENTLY BROKEN: Only first value captured and not as list
  result = run_cli(script, ["--ids", "1", "2", "3"], expect_error=False)
  assert result["stdout"].strip() == "ids=[1, 2, 3],names=[]"


def test_list_single_vs_multiple_values(run_cli):
  """Test distinction between single value and multiple values in List parameters.

  Documents current behavior for parsing single vs multiple values.
  CURRENTLY BROKEN: Neither single nor multiple values are handled correctly.
  """
  script = dedent("""
    from typing import List
    from params_proto import proto

    @proto.cli
    def main(values: List[int] = None):
        print(f"values={values}")

    if __name__ == "__main__":
        main()
    """)

  # Test 1: Single value should still be a list
  # CURRENTLY BROKEN: Returns "values=42" instead of "values=[42]"
  result = run_cli(script, ["--values", "42"], expect_error=False)
  assert result["stdout"].strip() == "values=[42]"

  # Test 2: Multiple values should all be in list
  # CURRENTLY BROKEN: Only first value captured, rest become positional args
  result = run_cli(script, ["--values", "1", "2", "3", "4", "5"], expect_error=False)
  assert result["stdout"].strip() == "values=[1, 2, 3, 4, 5]"


def test_list_help_strings(run_cli):
  """Test that List parameters have proper help text display.

  WORKS: Help generation works fine for List parameters.
  """
  script = dedent("""
    from typing import List
    from params_proto import proto

    @proto.cli
    def train(
        gpu_ids: List[int] = [0],  # GPU devices to use
        tags: List[str] = ["default"],  # Experiment tags
    ):
        '''Train with list parameters.'''
        pass

    if __name__ == "__main__":
        train()
    """)

  result = run_cli(script, ["--help"], expect_error=True)
  assert result["returncode"] == 0
  assert "--gpu-ids" in result["stdout"]
  assert "--tags" in result["stdout"]
  # Should have proper default display
  assert "[0]" in result["stdout"] or "[0, 1]" in result["stdout"]


def test_list_str_whitespace_handling(run_cli):
  """Test that List[str] properly handles strings with special characters.

  CURRENTLY BROKEN: Multiple values and proper list wrapping not implemented.
  """
  script = dedent("""
    from typing import List
    from params_proto import proto

    @proto.cli
    def main(paths: List[str] = None):
        print(f"paths={paths}")

    if __name__ == "__main__":
        main()
    """)

  # Test 1: Simple string paths
  # CURRENTLY BROKEN: Only first path captured, second becomes positional
  result = run_cli(script, ["--paths", "/home/user/data", "/mnt/dataset"], expect_error=False)
  assert "/home/user/data" in result["stdout"]
  assert "/mnt/dataset" in result["stdout"]

  # Test 2: Single path
  # CURRENTLY BROKEN: Path not wrapped in list
  result = run_cli(script, ["--paths", "./data"], expect_error=False)
  assert "./data" in result["stdout"]
