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

    --environment.domain STR     Domain name (e.g., cartpole, walker) (default: cartpole)
    --environment.task STR       Task name within the domain (default: swingup)
    --environment.time-limit FLOAT  Episode time limit in seconds (default: 10.0)

  Agent options:
    SAC agent hyperparameters.

    --agent.algorithm STR        RL algorithm (SAC or PPO) (default: SAC)
    --agent.buffer-size INT      Replay buffer capacity (default: 1000000)
    --agent.gamma FLOAT          Discount factor (default: 0.99)
    --agent.tau FLOAT            Target network update rate (default: 0.005)
  """)

  assert train_rl.__help_str__ == expected, "help string is not correct"


def test_proto_cli_bool_flags():
  """Test CLI help output for boolean flags.

  Boolean flags should show the form that changes the default:
  - default=False: show --flag (to enable)
  - default=True: show --no-flag (to disable)
  """
  from params_proto import proto

  @proto.cli(prog="train")
  def train(
    verbose: bool = False,  # Enable verbose output
    cuda: bool = True,  # Use CUDA acceleration
    debug: bool = False,
  ):
    """Train model with boolean configuration flags."""
    print(f"verbose={verbose}, cuda={cuda}")

  # Note: --no-cuda is shown because cuda defaults to True,
  # so the user would want --no-cuda to disable it
  expected = dedent("""
  usage: train [-h] [--verbose BOOL] [--no-cuda BOOL] [--debug BOOL]

  Train model with boolean configuration flags.

  options:
    -h, --help           show this help message and exit
    --verbose BOOL       Enable verbose output (default: False)
    --no-cuda BOOL       Use CUDA acceleration (default: True)
    --debug BOOL         Debug (default: False)
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
  usage: train [-h] [--checkpoint STR] [--resume-step INT] [--learning-rate FLOAT]

  Train model with optional checkpoint resume.

  options:
    -h, --help           show this help message and exit
    --checkpoint STR     Path to checkpoint file
    --resume-step INT    Step to resume from
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


# ===== CLI PARSING TESTS =====


def test_cli_parse_basic_arguments(monkeypatch):
  """Test CLI argument parsing with basic int and float types."""
  from params_proto import proto

  @proto.cli
  def train(
    seed: int,  # Random seed (required)
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Batch size
  ):
    """Train a model."""
    return {"seed": seed, "lr": lr, "batch_size": batch_size}

  # Mock sys.argv
  monkeypatch.setattr("sys.argv", ["train.py", "--seed", "42", "--lr", "0.01", "--batch-size", "64"])

  result = train()
  assert result["seed"] == 42
  assert result["lr"] == 0.01
  assert result["batch_size"] == 64


def test_cli_parse_required_parameter(monkeypatch):
  """Test that required parameters are enforced."""
  from params_proto import proto

  @proto.cli
  def train(
    seed: int,  # Required seed
    lr: float = 0.001,
  ):
    """Train a model."""
    return {"seed": seed, "lr": lr}

  # Missing required --seed should raise error
  monkeypatch.setattr("sys.argv", ["train.py", "--lr", "0.01"])

  with pytest.raises(SystemExit):
    train()


def test_cli_parse_defaults(monkeypatch):
  """Test that default values are used when not provided."""
  from params_proto import proto

  @proto.cli
  def train(
    seed: int = 42,
    lr: float = 0.001,
    batch_size: int = 32,
  ):
    """Train a model."""
    return {"seed": seed, "lr": lr, "batch_size": batch_size}

  # Only override seed, others use defaults
  monkeypatch.setattr("sys.argv", ["train.py", "--seed", "100"])

  result = train()
  assert result["seed"] == 100
  assert result["lr"] == 0.001
  assert result["batch_size"] == 32


def test_cli_parse_underscore_conversion(monkeypatch):
  """Test that underscores in parameter names are converted to hyphens in CLI."""
  from params_proto import proto

  @proto.cli
  def train(
    learning_rate: float = 0.001,
    batch_size: int = 32,
    num_epochs: int = 10,
  ):
    """Train a model."""
    return {"learning_rate": learning_rate, "batch_size": batch_size, "num_epochs": num_epochs}

  # Use hyphenated names in CLI
  monkeypatch.setattr("sys.argv", ["train.py", "--learning-rate", "0.01", "--batch-size", "64", "--num-epochs", "20"])

  result = train()
  assert result["learning_rate"] == 0.01
  assert result["batch_size"] == 64
  assert result["num_epochs"] == 20


def test_cli_parse_boolean_flags(monkeypatch):
  """Test boolean flag parsing (--flag / --no-flag)."""
  from params_proto import proto

  @proto.cli
  def train(
    verbose: bool = False,
    use_cuda: bool = True,
    debug: bool = False,
  ):
    """Train a model."""
    return {"verbose": verbose, "use_cuda": use_cuda, "debug": debug}

  # Test --flag sets to True
  monkeypatch.setattr("sys.argv", ["train.py", "--verbose", "--debug"])
  result = train()
  assert result["verbose"] == True
  assert result["use_cuda"] == True  # Default
  assert result["debug"] == True

  # Test --no-flag sets to False
  monkeypatch.setattr("sys.argv", ["train.py", "--no-use-cuda"])
  result = train()
  assert result["verbose"] == False  # Default
  assert result["use_cuda"] == False
  assert result["debug"] == False  # Default


def test_cli_parse_help_exits(monkeypatch, capsys):
  """Test that --help prints help and exits without calling function."""
  from params_proto import proto

  called = []

  @proto.cli
  def train(seed: int):
    """Train a model."""
    called.append(True)
    return {"seed": seed}

  # --help should exit without calling function
  monkeypatch.setattr("sys.argv", ["train.py", "--help"])

  with pytest.raises(SystemExit) as excinfo:
    train()

  # Should exit with code 0
  assert excinfo.value.code == 0

  # Function should NOT have been called
  assert len(called) == 0

  # Help text should be printed
  captured = capsys.readouterr()
  assert "usage:" in captured.out
  assert "--seed" in captured.out


def test_cli_programmatic_call_bypasses_argv(monkeypatch):
  """Test that calling with kwargs bypasses CLI parsing."""
  from params_proto import proto

  @proto.cli
  def train(
    seed: int,
    lr: float = 0.001,
  ):
    """Train a model."""
    return {"seed": seed, "lr": lr}

  # Set sys.argv to something else
  monkeypatch.setattr("sys.argv", ["train.py", "--seed", "999"])

  # Call with kwargs should use kwargs, not sys.argv
  result = train(seed=42, lr=0.01)
  assert result["seed"] == 42  # From kwargs, not sys.argv
  assert result["lr"] == 0.01


def test_cli_parse_positional_argument(monkeypatch):
  """Test that required parameters can be passed as positional arguments."""
  from params_proto import proto

  @proto.cli
  def train(
    seed: int,  # Required seed
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Batch size
  ):
    """Train a model."""
    return {"seed": seed, "lr": lr, "batch_size": batch_size}

  # Pass seed as positional argument
  monkeypatch.setattr("sys.argv", ["train.py", "42"])

  result = train()
  assert result["seed"] == 42
  assert result["lr"] == 0.001  # Default
  assert result["batch_size"] == 32  # Default


def test_cli_parse_positional_with_named_args(monkeypatch):
  """Test mixing positional and named arguments."""
  from params_proto import proto

  @proto.cli
  def train(
    seed: int,  # Required seed
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Batch size
  ):
    """Train a model."""
    return {"seed": seed, "lr": lr, "batch_size": batch_size}

  # Pass seed as positional, others as named
  monkeypatch.setattr("sys.argv", ["train.py", "42", "--lr", "0.01", "--batch-size", "64"])

  result = train()
  assert result["seed"] == 42
  assert result["lr"] == 0.01
  assert result["batch_size"] == 64


def test_cli_parse_named_overrides_positional(monkeypatch):
  """Test that named arguments can still be used even when positional is available."""
  from params_proto import proto

  @proto.cli
  def train(
    seed: int,  # Required seed
    lr: float = 0.001,
  ):
    """Train a model."""
    return {"seed": seed, "lr": lr}

  # Use named argument instead of positional
  monkeypatch.setattr("sys.argv", ["train.py", "--seed", "99"])

  result = train()
  assert result["seed"] == 99
  assert result["lr"] == 0.001


def test_cli_parse_multiple_required_positional(monkeypatch):
  """Test multiple required parameters as positional arguments."""
  from params_proto import proto

  @proto.cli
  def train(
    seed: int,  # Required seed
    num_epochs: int,  # Required epochs
    lr: float = 0.001,  # Learning rate
  ):
    """Train a model."""
    return {"seed": seed, "num_epochs": num_epochs, "lr": lr}

  # Pass both required params as positional
  monkeypatch.setattr("sys.argv", ["train.py", "42", "100"])

  result = train()
  assert result["seed"] == 42
  assert result["num_epochs"] == 100
  assert result["lr"] == 0.001


def test_cli_parse_positional_missing_required(monkeypatch):
  """Test that error is raised when required positional arg is missing."""
  from params_proto import proto

  @proto.cli
  def train(
    seed: int,  # Required seed
    lr: float = 0.001,
  ):
    """Train a model."""
    return {"seed": seed, "lr": lr}

  # Missing required seed parameter
  monkeypatch.setattr("sys.argv", ["train.py", "--lr", "0.01"])

  with pytest.raises(SystemExit):
    train()
