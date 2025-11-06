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


def test_inline_comments():
  """Test inline comments on the same line as parameter."""
  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    batch_size: int = 128,  # Training batch size
    learning_rate: float = 0.001,  # Initial learning rate
    epochs: int = 10,  # Number of training epochs
  ):
    """Train a model."""
    pass

  expected = dedent("""
  usage: train [-h] [--batch-size INT] [--learning-rate FLOAT] [--epochs INT]

  Train a model.

  options:
    -h, --help           show this help message and exit
    --batch-size INT     Training batch size (default: 128)
    --learning-rate FLOAT
                         Initial learning rate (default: 0.001)
    --epochs INT         Number of training epochs (default: 10)
  """)
  assert train.__help_str__ == expected, "inline comments should be extracted"


def test_line_above_comments():
  """Test comments on the line above parameter."""
  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    # Training batch size
    batch_size: int = 128,
    # Initial learning rate
    learning_rate: float = 0.001,
    # Number of training epochs
    epochs: int = 10,
  ):
    """Train a model."""
    pass

  # Line-above comments are NOT currently supported
  # They should be ignored
  expected = dedent("""
  usage: train [-h] [--batch-size INT] [--learning-rate FLOAT] [--epochs INT]

  Train a model.

  options:
    -h, --help           show this help message and exit
    --batch-size INT     Batch size (default: 128)
    --learning-rate FLOAT
                         Learning rate (default: 0.001)
    --epochs INT         Number of training epochs (default: 10)
  """)
  assert train.__help_str__ == expected, "line-above comments should be ignored"


def test_mixed_comments():
  """Test mix of inline and line-above comments."""
  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    batch_size: int = 128,  # Training batch size (inline)
    # This comment is ignored
    learning_rate: float = 0.001,
    epochs: int = 10,  # Number of epochs (inline)
  ):
    """Train a model."""
    pass

  expected = dedent("""
  usage: train [-h] [--batch-size INT] [--learning-rate FLOAT] [--epochs INT]

  Train a model.

  options:
    -h, --help           show this help message and exit
    --batch-size INT     Training batch size (inline) (default: 128)
    --learning-rate FLOAT
                         Learning rate (default: 0.001)
    --epochs INT         Number of epochs (inline) (default: 10)
  """)
  assert train.__help_str__ == expected, "only inline comments should be used"


def test_docstring_args_section():
  """Test that inline comments are concatenated with Args section in docstring."""
  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    batch_size: int = 128,  # Training batch size
    learning_rate: float = 0.001,  # Initial learning rate
    epochs: int = 10,
  ):
    """Train a model.

    Args:
      batch_size: Controls memory usage and gradient noise
      learning_rate: For the optimizer
      epochs: Number of training epochs
    """
    pass

  expected = dedent("""
  usage: train [-h] [--batch-size INT] [--learning-rate FLOAT] [--epochs INT]

  Train a model.

  options:
    -h, --help           show this help message and exit
    --batch-size INT     Training batch size. Controls memory usage and gradient noise (default: 128)
    --learning-rate FLOAT
                         Initial learning rate. For the optimizer (default: 0.001)
    --epochs INT         Number of training epochs (default: 10)
  """)
  assert train.__help_str__ == expected, (
    "inline comments and docstring Args should be concatenated"
  )


def test_no_comments():
  """Test parameters without any comments."""
  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    batch_size: int = 128,
    learning_rate: float = 0.001,
    epochs: int = 10,
  ):
    """Train a model."""
    pass

  expected = dedent("""
  usage: train [-h] [--batch-size INT] [--learning-rate FLOAT] [--epochs INT]

  Train a model.

  options:
    -h, --help           show this help message and exit
    --batch-size INT     Batch size (default: 128)
    --learning-rate FLOAT
                         Learning rate (default: 0.001)
    --epochs INT         Number of training epochs (default: 10)
  """)
  assert train.__help_str__ == expected, "auto-generated descriptions should be used"


def test_multiline_inline_comment():
  """Test that only first line of inline comment is used."""
  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    batch_size: int = 128,  # Training batch size
    # This continuation is ignored
    learning_rate: float = 0.001,
  ):
    """Train a model."""
    pass

  expected = dedent("""
  usage: train [-h] [--batch-size INT] [--learning-rate FLOAT]

  Train a model.

  options:
    -h, --help           show this help message and exit
    --batch-size INT     Training batch size (default: 128)
    --learning-rate FLOAT
                         Learning rate (default: 0.001)
  """)
  assert train.__help_str__ == expected, "only first line of inline comment used"
