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


def test_proto_basic_cli():
  """Test simple CLI help output for MNIST training."""
  from params_proto import proto

  @proto.cli(prog="train_mnist.py")
  def train_mnist(
    batch_size: int = 128,  # Training batch size
    epochs: int = 10,
  ):
    """Train an MLP on MNIST dataset.

    Args:
      batch_size: Training batch size
      epochs: Number of training epochs
    """
    print(f"Training MNIST with batch_size={batch_size}")

  expected = dedent("""
  usage: train_mnist.py [-h] [--batch-size INT] [--epochs INT]

  Train an MLP on MNIST dataset.

  options:
    -h, --help           show this help message and exit
    --batch-size INT     Training batch size (default: 128)
    --epochs INT         Number of training epochs (default: 10)
  """)
  assert train_mnist.__help_str__ == expected, "help string is not correct"


def test_proto_cli_long_name():
  """Test CLI help output with long parameter names."""
  from params_proto import proto

  @proto.cli(prog="train_rl")
  def train_rl(
    seed: int = 0,
    this_is_a_very_long_parameter_name: int = 1000000,
    this_is_an_even_longer_long_parameter_name: int = 1000000,
  ):
    """Train RL agent on dm_control environment.

    Args:
      seed: Random seed
      this_is_a_very_long_parameter_name: Very long parameter description
      this_is_an_even_longer_long_parameter_name: Even longer long parameter description
    """
    print(f"seed={seed} is")

  expected = dedent("""
  usage: train_rl [-h] [--seed INT] [--this-is-a-very-long-parameter-name INT] [--this-is-an-even-longer-long-parameter-name INT]

  Train RL agent on dm_control environment.

  options:
    -h, --help           show this help message and exit
    --seed INT           Random seed (default: 0)
    --this-is-a-very-long-parameter-name INT
                         Very long parameter description (default: 1000000)
    --this-is-an-even-longer-long-parameter-name INT
                         Even longer long parameter description (default: 1000000)
  """)
  assert train_rl.__help_str__ == expected, (
    "we should be able to handle long parameter names"
  )


def test_proto_cli_hierarchy():
  """Test CLI help output with @proto.prefix for hierarchical configs."""
  from params_proto import proto

  # @proto.prefix creates singleton config classes with CLI prefixes
  @proto.prefix
  class Environment:
    """dm_control environment configuration."""

    domain: str = "cartpole"  # Domain name (e.g., cartpole, walker)
    task: str = "swingup"  # Task name within the domain
    time_limit: float = 10.0  # Episode time limit in seconds

  @proto.prefix
  class Agent:
    """SAC agent hyperparameters."""

    algorithm: str = "SAC"  # RL algorithm (SAC or PPO)
    buffer_size: int = 1000000  # Replay buffer capacity
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate

  # @proto.cli is the entry point - called only once
  @proto.cli(prog="train_rl")
  def train_rl(
    total_steps: int = 1000000,  # Total environment steps
    eval_freq: int = 5000,  # Evaluation frequency
    seed: int = 0,
  ):
    """Train RL agent on dm_control environment."""
    print(f"Training {Agent.algorithm} on {Environment.domain}-{Environment.task}")

  # All decorators (@proto.cli, @proto.prefix, @proto) add:
  # - __init__(...args, **kwargs) method
  # - __repr__ for nice printing
  # - CLI argument parsing (for @proto.cli)
  expected = dedent("""
  usage: train_rl [-h] [--total-steps INT] [--eval-freq INT] [--seed INT] [OPTIONS]

  Train RL agent on dm_control environment.

  options:
    -h, --help                   show this help message and exit
    --total-steps INT            Total environment steps (default: 1000000)
    --eval-freq INT              Evaluation frequency (default: 5000)
    --seed INT                   Random seed (default: 0)

  Environment options:
    dm_control environment configuration.

    --Environment.domain STR     Domain name (e.g., cartpole, walker) (default: cartpole)
    --Environment.task STR       Task name within the domain (default: swingup)
    --Environment.time-limit FLOAT  Episode time limit in seconds (default: 10.0)

  Agent options:
    SAC agent hyperparameters.

    --Agent.algorithm STR        RL algorithm (SAC or PPO) (default: SAC)
    --Agent.buffer-size INT      Replay buffer capacity (default: 1000000)
    --Agent.gamma FLOAT          Discount factor (default: 0.99)
    --Agent.tau FLOAT            Target network update rate (default: 0.005)
  """)

  assert train_rl.__help_str__ == expected, "help string is not correct"


def test_proto_cli_bool_flags():
  """Test CLI help output for boolean flags."""
  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    verbose: bool = False,  # Enable verbose output
    cuda: bool = True,  # Use CUDA acceleration
    debug: bool = False,
  ):
    """Train model with boolean configuration flags."""
    print(f"verbose={verbose}, cuda={cuda}")

  expected = dedent("""
  usage: train [-h] [--verbose] [--cuda] [--debug]

  Train model with boolean configuration flags.

  options:
    -h, --help           show this help message and exit
    --verbose            Enable verbose output (default: False)
    --cuda               Use CUDA acceleration (default: True)
    --debug              Debug (default: False)
  """)
  assert train.__help_str__ == expected, "help string is not correct"


def test_proto_cli_optional_types():
  """Test CLI help output for Optional parameters."""
  from typing import Optional

  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    checkpoint: Optional[str] = None,  # Path to checkpoint file
    resume_step: Optional[int] = None,  # Step to resume from
    learning_rate: float = 0.001,
  ):
    """Train model with optional checkpoint resume."""
    print(f"checkpoint={checkpoint}, resume_step={resume_step}")

  expected = dedent("""
  usage: train [-h] [--checkpoint VALUE] [--resume-step VALUE] [--learning-rate FLOAT]

  Train model with optional checkpoint resume.

  options:
    -h, --help           show this help message and exit
    --checkpoint VALUE   Path to checkpoint file
    --resume-step VALUE  Step to resume from
    --learning-rate FLOAT
                         Learning rate (default: 0.001)
  """)
  assert train.__help_str__ == expected, "help string is not correct"


def test_proto_cli_list_types():
  """Test CLI help output for List parameters."""
  from typing import List

  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    gpu_ids: List[int] = [0, 1],  # GPU device IDs to use
    data_dirs: List[str] = ["./data"],  # Directories containing training data
    learning_rates: List[float] = [0.001, 0.0001],
  ):
    """Train model with list-based configuration."""
    print(f"gpu_ids={gpu_ids}")

  expected = dedent("""
  usage: train [-h] [--gpu-ids VALUE] [--data-dirs VALUE] [--learning-rates VALUE]

  Train model with list-based configuration.

  options:
    -h, --help           show this help message and exit
    --gpu-ids VALUE      GPU device IDs to use (default: [0, 1])
    --data-dirs VALUE    Directories containing training data (default: ['./data'])
    --learning-rates VALUE
                         Number of training learning rates (default: [0.001, 0.0001])
  """)
  assert train.__help_str__ == expected, "help string is not correct"


def test_proto_cli_literal_types():
  """Test CLI help output for Literal type choices."""
  from typing import Literal

  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    activation: Literal["relu", "gelu", "tanh"] = "relu",  # Activation function
    optimizer: Literal["adam", "sgd", "adamw"] = "adam",  # Optimizer algorithm
    precision: Literal["fp32", "fp16", "bf16"] = "fp32",
  ):
    """Train model with fixed choice parameters."""
    print(f"activation={activation}, optimizer={optimizer}")

  expected = dedent("""
  usage: train [-h] [--activation VALUE] [--optimizer VALUE] [--precision VALUE]

  Train model with fixed choice parameters.

  options:
    -h, --help           show this help message and exit
    --activation VALUE   Activation function (default: relu)
    --optimizer VALUE    Optimizer algorithm (default: adam)
    --precision VALUE    Precision (default: fp32)
  """)
  assert train.__help_str__ == expected, "help string is not correct"


def test_proto_cli_tuple_types():
  """Test CLI help output for Tuple parameters."""
  from typing import Tuple

  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    image_size: Tuple[int, int] = (224, 224),  # Image height and width
    norm_mean: Tuple[float, float, float] = (
      0.485,
      0.456,
      0.406,
    ),  # RGB normalization mean
    window_size: tuple[int, int] = (7, 7),
  ):
    """Train model with fixed-length tuple configuration."""
    print(f"image_size={image_size}")

  expected = dedent("""
  usage: train [-h] [--image-size VALUE] [--norm-mean VALUE] [--window-size VALUE]

  Train model with fixed-length tuple configuration.

  options:
    -h, --help           show this help message and exit
    --image-size VALUE   Image height and width (default: (224, 224))
    --norm-mean VALUE    Norm mean (default: (0.485, 0.456, 0.406))
    --window-size VALUE  Window size (default: (7, 7))
  """)
  assert train.__help_str__ == expected, "help string is not correct"


def test_proto_cli_enum_types():
  """Test CLI help output for Enum types."""
  from enum import Enum

  from params_proto import proto

  class Optimizer(Enum):
    """Optimizer algorithms."""

    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"

  class Precision(Enum):
    """Training precision modes."""

    FP32 = 32
    FP16 = 16
    BF16 = "bf16"

  @proto.cli(prog="train")
  def train(
    optimizer: Optimizer = Optimizer.ADAM,  # Optimizer algorithm
    precision: Precision = Precision.FP32,  # Training precision mode
    learning_rate: float = 0.001,
  ):
    """Train model with enum-based configuration."""
    print(f"optimizer={optimizer.value}")

  expected = dedent("""
  usage: train [-h] [--optimizer {ADAM,SGD,ADAMW}] [--precision {FP32,FP16,BF16}] [--learning-rate FLOAT]

  Train model with enum-based configuration.

  options:
    -h, --help           show this help message and exit
    --optimizer {ADAM,SGD,ADAMW}
                         Optimizer algorithm (default: Optimizer.ADAM)
    --precision {FP32,FP16,BF16}
                         Training precision mode (default: Precision.FP32)
    --learning-rate FLOAT
                         Learning rate (default: 0.001)
  """)
  assert train.__help_str__ == expected, "help string is not correct"


def test_proto_cli_path_types():
  """Test CLI help output for Path parameters."""
  from pathlib import Path

  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    output_dir: Path = Path("./outputs"),  # Output directory for results
    checkpoint_path: Path = Path("./checkpoints/model.pt"),  # Path to checkpoint file
    data_root: Path = Path("/data"),
  ):
    """Train model with Path-based file system configuration."""
    print(f"output_dir={output_dir}")

  expected = dedent("""
  usage: train [-h] [--output-dir VALUE] [--checkpoint-path VALUE] [--data-root VALUE]

  Train model with Path-based file system configuration.

  options:
    -h, --help           show this help message and exit
    --output-dir VALUE   Output directory for results (default: outputs)
    --checkpoint-path VALUE
                         Path to checkpoint file (default: checkpoints/model.pt)
    --data-root VALUE    Data root (default: /data)
  """)
  assert train.__help_str__ == expected, "help string is not correct"


def test_proto_cli_union_types():
  """Test CLI help output for Union type parameters."""
  from typing import Union

  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    learning_rate: Union[float, str] = 0.001,  # Learning rate or 'auto'
    batch_size: Union[int, str] = 32,  # Batch size or 'auto'
    seed: int = 42,
  ):
    """Train model with Union type flexibility."""
    print(f"learning_rate={learning_rate}, batch_size={batch_size}")

  expected = dedent("""
  usage: train [-h] [--learning-rate VALUE] [--batch-size VALUE] [--seed INT]

  Train model with Union type flexibility.

  options:
    -h, --help           show this help message and exit
    --learning-rate VALUE
                         Learning rate or 'auto' (default: 0.001)
    --batch-size VALUE   Batch size or 'auto' (default: 32)
    --seed INT           Random seed (default: 42)
  """)
  assert train.__help_str__ == expected, "help string is not correct"
