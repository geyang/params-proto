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
  """Test that comments on the line above parameter are ignored.

  Line-above comments should NOT be extracted. This test proves they are
  ignored by showing that auto-generated descriptions are used instead.
  """
  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    # This line-above comment should be IGNORED
    batch_size: int = 128,
    # This line-above comment should also be IGNORED
    learning_rate: float = 0.001,
    # This should be IGNORED too
    epochs: int = 10,
  ):
    """Train a model."""
    pass

  # If line-above comments were extracted, we'd see them in the help text
  # Instead, we should see auto-generated descriptions
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
  assert train.__help_str__ == expected, "line-above comments should be ignored; auto-generated descriptions should be used"


def test_mixed_comments():
  """Test that line-above comments are ignored even when mixed with inline comments.

  Proves that:
  - Inline comments ARE extracted
  - Line-above comments are IGNORED (auto-generated description used instead)
  """
  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    batch_size: int = 128,  # Training batch size (inline - USED)
    # This line-above comment should be IGNORED, not used
    learning_rate: float = 0.001,
    epochs: int = 10,  # Number of epochs (inline - USED)
  ):
    """Train a model."""
    pass

  # Note: learning_rate has "Learning rate" (auto-generated), NOT "This line-above comment..."
  expected = dedent("""
  usage: train [-h] [--batch-size INT] [--learning-rate FLOAT] [--epochs INT]

  Train a model.

  options:
    -h, --help           show this help message and exit
    --batch-size INT     Training batch size (inline - USED) (default: 128)
    --learning-rate FLOAT
                         Learning rate (default: 0.001)
    --epochs INT         Number of epochs (inline - USED) (default: 10)
  """)
  assert train.__help_str__ == expected, "only inline comments should be used; line-above comments must be ignored"


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
    --batch-size INT     Training batch size
                         Controls memory usage and gradient noise (default: 128)
    --learning-rate FLOAT
                         Initial learning rate
                         For the optimizer (default: 0.001)
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


def test_inline_comment_continuation_ignored():
  """Test that comment continuations on following lines are ignored.

  Only the inline comment on the same line as the parameter is extracted.
  Comments on subsequent lines (even if they look like continuations) are NOT extracted.
  """
  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    batch_size: int = 128,  # Training batch size
    # This line looks like a continuation but is IGNORED
    learning_rate: float = 0.001,
  ):
    """Train a model."""
    pass

  # learning_rate gets auto-generated "Learning rate", NOT the line-above comment
  expected = dedent("""
  usage: train [-h] [--batch-size INT] [--learning-rate FLOAT]

  Train a model.

  options:
    -h, --help           show this help message and exit
    --batch-size INT     Training batch size (default: 128)
    --learning-rate FLOAT
                         Learning rate (default: 0.001)
  """)
  assert train.__help_str__ == expected, "comment continuations on next lines should be ignored"
