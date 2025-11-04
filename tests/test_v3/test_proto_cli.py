from textwrap import dedent


def test_proto_cli():
  """Test simple CLI help output for MNIST training."""
  from params_proto import proto

  @proto.cli
  def train_mnist(
    batch_size: int = 128,  # Training batch size
    epochs: int = 10,
  ):
    """Train an MLP on MNIST dataset."""
    print(f"Training MNIST with batch_size={batch_size}")

  # @proto.cli decorator adds __init__, __repr__, and CLI parsing capabilities
  expected = dedent("""
  usage: mnist_train.py [-h] [--batch-size INT] [--epochs INT]

  Train an MLP on MNIST dataset.

  options:
    -h, --help           show this help message and exit
    --batch-size INT     Training batch size (default: 128)
    --epochs INT         Number of training epochs (default: 10)
  """)
  assert train_mnist.__help_str__ == expected, "help string is not correct"


def test_rl_agent_help():
  """Test help output for RL agent with dm_control environment."""
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
  @proto.cli
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
  usage: train_rl.py [-h] [--total-steps INT] [--eval-freq INT] [--seed INT] [OPTIONS]

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
