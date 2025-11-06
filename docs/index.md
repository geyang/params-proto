# Avoiding Hyperparameter Hell with `params-proto`

**params-proto v3**
is a declarative hyperparameter management library for machine learning. Write your parameters once with type hints and
inline comments to get automatic CLI parsing, help generation, and declarative parameter sweeps with explicit error
messages.

- Automatically parse type hints and inline comments into an CLI program
- Your IDE will provide autocompletion and type checking for your parameters
- As simple as a class namespace or a function, progressively build up to more complex programs.
- Multiple override patterns: CLI, direct assignment, context managers, and yaml config files

## Quick Start

Install params-proto using uv or pip:

```shell
uv add params-proto
```

or

```shell
pip install params-proto
```

### Basic Function CLI

Convert any Python function into a CLI program with a single decorator:

```python
from params_proto import proto


@proto.cli
def train_mnist(
    batch_size: int = 128,  # Training batch size
    epochs: int = 10,  # Number of training epochs
):
    """Train an MLP on MNIST dataset.

    Args:
        batch_size: Training batch size
        epochs: Number of training epochs
    """
    print(f"Training MNIST with batch_size={batch_size}")


if __name__ == "__main__":
    train_mnist()
```

Running `python train_mnist.py --help` gives you:

```
usage: train_mnist.py [-h] [--batch-size INT] [--epochs INT]

Train an MLP on MNIST dataset.

options:
  -h, --help           show this help message and exit
  --batch-size INT     Training batch size (default: 128)
  --epochs INT         Number of training epochs (default: 10)
```

Override parameters from the command line:

```shell
$ python train_mnist.py --batch-size 256 --epochs 20
Training MNIST with batch_size=256
```

### Hierarchical Configuration

Use `@proto.prefix` to create organized, hierarchical configs:

```python
from params_proto import proto


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


@proto.cli
def train_rl(
    total_steps: int = 1000000,  # Total environment steps
    eval_freq: int = 5000,  # Evaluation frequency
    seed: int = 0,  # Random seed
):
    """Train RL agent on dm_control environment."""
    print(f"Training {Agent.algorithm} on {Environment.domain}-{Environment.task}")


if __name__ == "__main__":
    train_rl()
```

Running `python train_rl.py --help` shows grouped options:

```
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
```

Override any parameter:

```shell
$ python train_rl.py --Environment.domain walker --Agent.gamma 0.95
Training SAC on walker-swingup
```

## Documentation Contents

```{toctree}
:maxdepth: 1
:caption: Getting Started

quick_start
migration
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guide/decorators
guide/functions
guide/classes
guide/types
guide/overrides
guide/prefixes
```

```{toctree}
:maxdepth: 1
:caption: Examples

examples/basic_usage
examples/ml_training
examples/rl_agent
examples/cli_applications
```

```{toctree}
:maxdepth: 1
:caption: API Reference

api/proto
api/utils
```

```{toctree}
:maxdepth: 1
:caption: Additional Resources

release_notes
```

## The Parameter Hell and Other Anti-Patterns

When is the last time you spent hours tracking down a config variable, jumping
through layers of class inheritance and runtime values? Only to find out that
the value you were looking for was never set?

Good code should be self-explanatory and statically resolvable. You should be able
to take a look at the code and understand what value the parameter will take. 
A common anti-pattern that appears everywhere is to nest config classes, and pass
a single `config` object to the constructure of the top-level class. This makes it 
difficult to **connect config flags with where and how it is used**. Take a look
at the example below:

- **Nested Config Classes Without Type Hint**: A common anti-pattern is to nest
  config classes, and pass a single `config` object to the constructor of the
  top-level class.

  ```python
  class Impala:
  def __init__(self, config: Config):
      self.config = config

      # All of these uses the same config!
      self.critic = Critic(self.config)
      self.actor = Actor(self.config)
      # ...
  
  class Critic:
  def __init__(self, config: Config):
      self.config = config
      self.lr = config.lr
      self.ent_coef = config.critic.ent_coef
      self.discount = config.critic.discount
      # ...
  
  def loss(self, x):
      if self.config.loss_type == "l2":
        return nn.l2_loss(x) # ...
  
  
  ```

  This tends to make it impossible to understand what parameter is used where.

## Single-Source-Of-Truth HyperParameter Management with `params-proto`

For a simple MNIST example we can define the training parameters by the
function signature of `def train(lr: float...):`, and `def eval(batch_size: int...):`.
And for the overall training run parameters, we can have those defined by the entrypoint
function `def main(seed: int...):`

This way, your IDE can provide definition/usage intellisense, and you can
easily see what parameters are used where. The idea is to **co-locate parameters
with their usages** throughout the codebase.

### When is Centralized Configuration Useful?

It is helpful when we only need to look at one place to override default values, for
example at the beginning of your training run. `params-proto` does so by keeping track
of all configuration parameters in a single place. So to automatically override
parameters, you can simply run

```python
from params_proto import proto

@proto.cli
def train(
    lr: float = 0.001,  # Learning rate
    batch_size: int = 32,  # Training batch size
    n_epochs: int = 100,  # Number of training epochs
):
    """Train a neural network on CIFAR-10 dataset."""
    print(f"Training with lr={lr}, batch_size={batch_size}, n_epochs={n_epochs}")
    
dep = {"train.lr": 0.01, "train.batch_size": 256, "train.n_epochs": 10}

proto.bind(**dep: DepencencyDict)

assert train.lr == 0.01, "this is now overriden."
assert train.batch_size == 256, "this is now overriden."
```

- Track what parameters exist
- Get IDE autocompletion and type checking
- Change parameters from the command line
- Maintain parameter documentation

params-proto solves this by providing a declarative way to define parameters that integrates seamlessly with Python IDEs
and command-line interfaces.

## Why v3?

`params-proto` v3 is a complete redesign focused on simplicity, that maximally takes
advantage of Python's new type hint system. This new API is in fact quite similar to
`params-proto` v1, but at that time when I developed v1 (back in 2018), the python
type hint system was still in its infancy. v2 was a rewrite from v1 that introduced
the `params_proto.hyper` module, that allowed us to create and load hyperparameter
sweeps. 

v2 simply returns us to that, but with a modern, polished, and unified API for both
decorator-based cli programs and the same powerful parameter sweeps.

| Feature     | v1                | v2                | v3           |
|-------------|-------------------|-------------------|--------------|
| API Style   | Decorators        | Class inheritance | Decorators   |
| Type Hints  | Not available     | Optional          | Required     |
| Inline Docs | Manual            | Manual            | Automatic    |
| Functions   | Full support      | Not supported     | Full support |
| Union Types | Not supported     | Limited           | Full support |
| IDE Support | Basic             | Basic             | Excellent    |

See the [Migration Guide](migration.md) for upgrading from v2.

## GitHub Repository

The source code is available on [GitHub](https://github.com/geyang/params-proto).

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
