"""Tests for required parameters in CLI help text."""

from textwrap import dedent

import pytest


@pytest.fixture(autouse=True)
def clear_proto_state():
  """Clear global proto state before each test."""
  import sys
  import params_proto.proto  # Ensure module is loaded

  # Get the actual proto module (not the decorator function)
  proto_module = sys.modules["params_proto.proto"]
  # Clear the global registries
  proto_module._SINGLETONS.clear()
  proto_module._BIND_CONTEXT.clear()
  if hasattr(proto_module, "_BIND_STACK"):
    proto_module._BIND_STACK.clear()

  yield

  # Clean up after test too
  proto_module._SINGLETONS.clear()
  proto_module._BIND_CONTEXT.clear()
  if hasattr(proto_module, "_BIND_STACK"):
    proto_module._BIND_STACK.clear()


def test_required_parameter():
  """Test that parameters without defaults show (required)."""
  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    data_path: str,  # Path to training data
    batch_size: int = 32,  # Batch size
  ):
    """Train a model."""
    pass

  expected = dedent("""
  usage: train [-h] [--data-path STR] [--batch-size INT]

  Train a model.

  options:
    -h, --help           show this help message and exit
    --data-path STR      Path to training data (required)
    --batch-size INT     Batch size (default: 32)
  """)

  assert train.__help_str__ == expected


def test_all_required_parameters():
  """Test multiple required parameters."""
  from params_proto import proto

  @proto.cli(prog="convert")
  def convert(
    input_file: str,  # Input file path
    output_file: str,  # Output file path
    format: str,  # Output format
  ):
    """Convert files between formats."""
    pass

  expected = dedent("""
  usage: convert [-h] [--input-file STR] [--output-file STR] [--format STR]

  Convert files between formats.

  options:
    -h, --help           show this help message and exit
    --input-file STR     Input file path (required)
    --output-file STR    Output file path (required)
    --format STR         Output format (required)
  """)

  assert convert.__help_str__ == expected


def test_all_optional_parameters():
  """Test that all optional parameters show defaults."""
  from params_proto import proto

  @proto.cli(prog="run")
  def run(
    lr: float = 0.001,  # Learning rate
    epochs: int = 10,  # Number of epochs
    seed: int = 42,  # Random seed
  ):
    """Run training."""
    pass

  expected = dedent("""
  usage: run [-h] [--lr FLOAT] [--epochs INT] [--seed INT]

  Run training.

  options:
    -h, --help           show this help message and exit
    --lr FLOAT           Learning rate (default: 0.001)
    --epochs INT         Number of epochs (default: 10)
    --seed INT           Random seed (default: 42)
  """)

  assert run.__help_str__ == expected


def test_mixed_required_and_optional():
  """Test mix of required and optional parameters."""
  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    model_name: str,  # Name of model architecture
    data_dir: str,  # Data directory
    output_dir: str = "./outputs",  # Output directory
    learning_rate: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Batch size
  ):
    """Train a model with the given configuration."""
    pass

  expected = dedent("""
  usage: train [-h] [--model-name STR] [--data-dir STR] [--output-dir STR] [--learning-rate FLOAT] [--batch-size INT]

  Train a model with the given configuration.

  options:
    -h, --help           show this help message and exit
    --model-name STR     Name of model architecture (required)
    --data-dir STR       Data directory (required)
    --output-dir STR     Output directory (default: ./outputs)
    --learning-rate FLOAT
                         Learning rate (default: 0.001)
    --batch-size INT     Batch size (default: 32)
  """)

  assert train.__help_str__ == expected


def test_required_with_multiline_doc():
  """Test required parameter with multi-line documentation."""
  from params_proto import proto

  @proto.cli(prog="process")
  def process(
    config_file: str,  # Configuration file path
    workers: int = 4,  # Number of workers
  ):
    """Process data in parallel.

    Args:
        config_file: Path to YAML configuration file. Must contain dataset
                    specifications and processing parameters.
        workers: Number of parallel worker processes to use.
    """
    pass

  expected = dedent("""
  usage: process [-h] [--config-file STR] [--workers INT]

  Process data in parallel.

  options:
    -h, --help           show this help message and exit
    --config-file STR    Configuration file path
                         Path to YAML configuration file. Must contain dataset specifications and processing parameters. (required)
    --workers INT        Number of workers
                         Number of parallel worker processes to use. (default: 4)
  """)

  assert process.__help_str__ == expected


def test_required_no_inline_comment():
  """Test required parameter gets auto-generated description."""
  from params_proto import proto

  @proto.cli(prog="init")
  def init(
    project_name: str,
    template: str = "basic",  # Project template
  ):
    """Initialize a new project."""
    pass

  expected = dedent("""
  usage: init [-h] [--project-name STR] [--template STR]

  Initialize a new project.

  options:
    -h, --help           show this help message and exit
    --project-name STR   Project name (required)
    --template STR       Project template (default: basic)
  """)

  assert init.__help_str__ == expected


def test_required_boolean_flag():
  """Test required boolean parameters."""
  from params_proto import proto

  @proto.cli(prog="build")
  def build(
    enable_optimizations: bool,  # Enable compiler optimizations
    verbose: bool = False,  # Verbose output
  ):
    """Build the project."""
    pass

  expected = dedent("""
  usage: build [-h] [--enable-optimizations BOOL] [--verbose BOOL]

  Build the project.

  options:
    -h, --help           show this help message and exit
    --enable-optimizations BOOL
                         Enable compiler optimizations (required)
    --verbose BOOL       Verbose output (default: False)
  """)

  assert build.__help_str__ == expected
