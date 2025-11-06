"""Tests for line breaking with long comments in CLI help text."""

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


def test_long_comment_no_wrapping_in_plain_text():
  """Test that __help_str__ does NOT wrap long comments (plain text for testing)."""
  from params_proto import proto

  @proto.cli(prog='train')
  def train(
    data_path: str,  # Path to the training data directory containing images, annotations, and metadata files for model training
    batch_size: int = 32,  # Batch size
  ):
    """Train a model."""
    pass

  # Plain __help_str__ should NOT wrap - it's for testing
  help_text = train.__help_str__

  # Check that the long line is NOT wrapped (stays on one line)
  assert "Path to the training data directory containing images, annotations, and metadata files for model training (required)" in help_text

  # The entire description should be on the same line as the parameter (or continuation)
  lines = help_text.split('\n')
  data_path_lines = [l for l in lines if '--data-path' in l or ('annotations' in l and 'metadata' in l)]

  # Check that it's all there (may be on continuation line if option string is long)
  help_section = '\n'.join(lines)
  assert "annotations" in help_section
  assert "metadata" in help_section
  assert "(required)" in help_section


def test_very_long_comments_preserved():
  """Test that very long comments are preserved in plain __help_str__."""
  from params_proto import proto

  @proto.cli(prog='process')
  def process(
    model_config: str,  # YAML configuration file specifying model architecture, hyperparameters, optimizer settings, and training schedule parameters
    output_dir: str = "./outputs",  # Directory where checkpoints, logs, and evaluation results will be saved during and after training
  ):
    """Process data."""
    pass

  help_text = process.__help_str__

  # Both long comments should be preserved in full
  assert "YAML configuration file specifying model architecture" in help_text
  assert "hyperparameters, optimizer settings, and training schedule parameters" in help_text

  assert "Directory where checkpoints, logs, and evaluation results" in help_text
  assert "will be saved during and after training" in help_text


def test_mixed_short_and_long_comments():
  """Test mix of short and long comments."""
  from params_proto import proto

  @proto.cli(prog='train')
  def train(
    very_long_param: str,  # This is a very long comment that describes the parameter in great detail and should probably be wrapped for terminal display but not in plain text
    short: int = 1,  # Short
    medium: float = 0.5,  # Medium length comment here
  ):
    """Train."""
    pass

  expected = dedent("""
  usage: train [-h] [--very-long-param STR] [--short INT] [--medium FLOAT]

  Train.

  options:
    -h, --help           show this help message and exit
    --very-long-param STR
                         This is a very long comment that describes the parameter in great detail and should probably be wrapped for terminal display but not in plain text (required)
    --short INT          Short (default: 1)
    --medium FLOAT       Medium length comment here (default: 0.5)
  """)

  assert train.__help_str__ == expected


def test_multiline_docstring_with_long_args():
  """Test that multi-line Args docstrings are preserved fully."""
  from params_proto import proto

  @proto.cli(prog='train')
  def train(
    config: str,  # Config file
    workers: int = 4,  # Workers
  ):
    """Train a model.

    Args:
        config: Path to YAML configuration file specifying all model hyperparameters, data augmentation settings, and training schedule parameters
        workers: Number of parallel worker processes to use during data loading and preprocessing
    """
    pass

  help_text = train.__help_str__

  # The concatenated inline + docstring should be preserved
  assert "Config file" in help_text
  assert "Path to YAML configuration file" in help_text
  assert "hyperparameters, data augmentation settings, and training schedule parameters" in help_text


def test_boolean_with_long_comment():
  """Test boolean flags with long comments."""
  from params_proto import proto

  @proto.cli(prog='build')
  def build(
    enable_optimizations: bool,  # Enable compiler optimizations including loop unrolling, dead code elimination, and constant folding
    verbose: bool = False,  # Verbose
  ):
    """Build."""
    pass

  help_text = build.__help_str__

  # Long boolean comment should be preserved
  assert "Enable compiler optimizations including loop unrolling" in help_text
  assert "dead code elimination, and constant folding" in help_text
  assert "(required)" in help_text


def test_extremely_long_single_line():
  """Test extremely long single-line comment (>120 chars)."""
  from params_proto import proto

  @proto.cli(prog='test')
  def test_func(
    param: str,  # This is an extremely long comment that goes on and on describing every possible detail about this parameter including its type, usage, constraints, default behavior, edge cases, and various other important considerations that the user absolutely must know
  ):
    """Test."""
    pass

  help_text = test_func.__help_str__

  # Should be preserved in full in plain text (for testing)
  assert "This is an extremely long comment" in help_text
  assert "various other important considerations" in help_text
  assert "that the user absolutely must know (required)" in help_text
