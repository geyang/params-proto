# Avoiding Hyperparameter Hell 🔥 with `params-proto`

**params-proto v3**
is a declarative hyperparameter management library for machine learning. Write your parameters once with type hints and
inline comments to get automatic CLI parsing, help generation, and declarative parameter sweeps with explicit error
messages.

- Automatically parse type hints and inline comments into an CLI program
- Your IDE will provide autocompletion and type checking for your parameters
- As simple as a class namespace or a function, progressively build up to more complex programs.
- Multiple override patterns: CLI, direct assignment, context managers, and yaml config files.

Here is a quick example: first install params-proto using uv or pip

```shell
uv add params-proto=={VERSION}  # or
pip install params-proto=={VERSION}
```

Now you can convert this function into a CLI program:

```python
from params_proto import proto


@proto.cli
def train_mnist(
  seed: int,  # Random seed
  batch_size: int = 128,  # Training batch size
  epochs: int = 10,  # Number of training epochs
):
  """Train an MLP on MNIST dataset."""

  print(f"yoooooo~ seed={seed}, batch_size={batch_size}")


if __name__ == "__main__":
  train_mnist()
```

Running `python train_mnist.py --help` gives you colorized output in the terminal:

```{ansi-block}
:string_escape:

usage: train_mnist.py [-h] [--seed INT] [--batch-size INT] [--epochs INT]

Train an MLP on MNIST dataset.

options:
  -h, --help           show this help message and exit
  --seed \x1b[1m\x1b[94mINT\x1b[0m           Random seed \x1b[1m\x1b[31m(required)\x1b[0m
  --batch-size \x1b[1m\x1b[94mINT\x1b[0m     Training batch size \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m128\x1b[0m\x1b[36m)\x1b[0m
  --epochs \x1b[1m\x1b[94mINT\x1b[0m         Number of training epochs \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m10\x1b[0m\x1b[36m)\x1b[0m
```

## What is Parameter Hell 🔥? (and Other Anti-Patterns)

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


proto.bind(lr=0.01, batch_size=256, n_epochs=10)

assert train.lr == 0.01, "this is now overridden."
assert train.batch_size == 256, "this is now overridden."
assert train.n_epochs == 10, "this is now overridden."
```

- Track what parameters exist
- Get IDE autocompletion and type checking
- Change parameters from the command line
- Maintain parameter documentation

params-proto solves this by providing a declarative way to define parameters that integrates seamlessly with Python IDEs
and command-line interfaces.

Override parameters from the command line:

```shell
$ python train_mnist.py --batch-size 256 --epochs 20
Training MNIST with batch_size=256
```

### Composing Configs from Multiple Locations

Use `@proto.prefix` to organize configuration across multiple namespaces:

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

```{ansi-block}
:string_escape:

usage: train_rl.py [-h] [--total-steps \x1b[1m\x1b[94mINT\x1b[0m] [--eval-freq \x1b[1m\x1b[94mINT\x1b[0m] [--seed \x1b[1m\x1b[94mINT\x1b[0m] [OPTIONS]

Train RL agent on dm_control environment.

options:
  -h, --help                   show this help message and exit
  --total-steps \x1b[1m\x1b[94mINT\x1b[0m            Total environment steps \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m1000000\x1b[0m\x1b[36m)\x1b[0m
  --eval-freq \x1b[1m\x1b[94mINT\x1b[0m              Evaluation frequency \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m5000\x1b[0m\x1b[36m)\x1b[0m
  --seed \x1b[1m\x1b[94mINT\x1b[0m                   Random seed \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m0\x1b[0m\x1b[36m)\x1b[0m

Environment options:
  dm_control environment configuration.

  --Environment.domain \x1b[1m\x1b[94mSTR\x1b[0m     Domain name (e.g., cartpole, walker) \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36mcartpole\x1b[0m\x1b[36m)\x1b[0m
  --Environment.task \x1b[1m\x1b[94mSTR\x1b[0m       Task name within the domain \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36mswingup\x1b[0m\x1b[36m)\x1b[0m
  --Environment.time-limit \x1b[1m\x1b[94mFLOAT\x1b[0m  Episode time limit in seconds \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m10.0\x1b[0m\x1b[36m)\x1b[0m

Agent options:
  SAC agent hyperparameters.

  --Agent.algorithm \x1b[1m\x1b[94mSTR\x1b[0m        RL algorithm (SAC or PPO) \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36mSAC\x1b[0m\x1b[36m)\x1b[0m
  --Agent.buffer-size \x1b[1m\x1b[94mINT\x1b[0m      Replay buffer capacity \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m1000000\x1b[0m\x1b[36m)\x1b[0m
  --Agent.gamma \x1b[1m\x1b[94mFLOAT\x1b[0m          Discount factor \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m0.99\x1b[0m\x1b[36m)\x1b[0m
  --Agent.tau \x1b[1m\x1b[94mFLOAT\x1b[0m            Target network update rate \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m0.005\x1b[0m\x1b[36m)\x1b[0m
```

Override any parameter:

```shell
$ python train_rl.py --Environment.domain walker --Agent.gamma 0.95
Training SAC on walker-swingup
```

## Environment Variables

Read configuration from environment variables with type-safe defaults:

```python
from params_proto import proto, EnvVar


@proto.cli
def train_model(
  # Three ways to specify environment variables:
  batch_size: int = EnvVar @ "BATCH_SIZE",  # Read from env var
  learning_rate: float = EnvVar @ "LR" | 0.001,  # With default fallback
  db_url: str = EnvVar("DATABASE_URL", default="localhost"),  # Function syntax
  data_dir: str = EnvVar @ "$DATA_DIR/models",  # Template expansion
):
  """Train model with environment configuration."""
  print(f"Training with batch_size={batch_size}, lr={learning_rate}")


if __name__ == "__main__":
  train_model()
```

The pipe operator (`|`) provides clean syntax for fallback values:

```python
# If LR env var is set, use it; otherwise use 0.001
learning_rate: float = EnvVar @ "LR" | 0.001
```

Environment variables are resolved at decoration time and automatically converted to the annotated type (int, float, bool, str).

See **[Environment Variables Guide](key_concepts/environment_variables.md)** for comprehensive documentation including template expansion, security considerations, and common patterns.

## Learn More

The Quick Start covers the basics. For deeper understanding, see:

- **[Core Concepts](key_concepts/core-concepts.md)** - The three main decorators (@proto, @proto.cli, @proto.prefix)
- **[Configuration Patterns](key_concepts/configuration-patterns.md)** - Functions vs classes, when to use each

**Building CLIs** (related guides for creating command-line interfaces):
- **[CLI Fundamentals](key_concepts/cli-fundamentals.md)** - Basic CLI features and type display
- **[CLI Patterns](key_concepts/cli-patterns.md)** - Advanced patterns like grouped options and custom program names
- **[Naming Conventions](key_concepts/naming-conventions.md)** - How Python names convert to CLI arguments
- **[Help Generation](key_concepts/help-generation.md)** - Automatic help text from comments and docstrings

- **[Type System](key_concepts/type-system.md)** - Complete type hints reference and type validation
- **[Union Types: Subcommands](key_concepts/union_types.md)** - Union types as subcommands, optional parameters, and multi-way dispatching
- **[Environment Variables](key_concepts/environment_variables.md)** - Configuration from environment with EnvVar
- **[Parameter Overrides](key_concepts/parameter-overrides.md)** - CLI, context managers, and other override methods
- **[Advanced Patterns: Prefixes & Composition](key_concepts/advanced_patterns.md)** - Global singleton configs with @proto.prefix, namespaced parameters, and complex composition
- **[Hyperparameter Sweeps](key_concepts/hyperparameter_sweeps.md)** - Declarative parameter sweeps with Sweep
- **[Parameter Iteration](key_concepts/parameter-iteration.md)** - Lightweight, composable sweeps with piter
- **[ANSI Formatting](key_concepts/ansi_formatting.md)** - Terminal colors and formatting
- **[Claude Skill](key_concepts/claude_skill.md)** - AI assistance for params-proto development

## Documentation Contents

```{toctree}
:maxdepth: 1
:caption: Introduction

quick_start
migration
Release Notes <release_notes>
```

```{toctree}
:maxdepth: 2
:caption: Key Concepts

Welcome <key_concepts/welcome>
Core Concepts <key_concepts/core-concepts>
Configuration Patterns <key_concepts/configuration-patterns>
CLI Fundamentals <key_concepts/cli-fundamentals>
CLI Patterns <key_concepts/cli-patterns>
Naming Conventions <key_concepts/naming-conventions>
Help Generation <key_concepts/help-generation>
Type System <key_concepts/type-system>
Union Types: Subcommands <key_concepts/union_types>
Environment Variables <key_concepts/environment_variables>
Parameter Overrides <key_concepts/parameter-overrides>
Advanced Patterns: Prefixes & Composition <key_concepts/advanced_patterns>
Hyperparameter Sweeps <key_concepts/hyperparameter_sweeps>
Parameter Iteration <key_concepts/parameter-iteration>
ANSI Formatting <key_concepts/ansi_formatting>
Claude Skill <key_concepts/claude_skill>
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

## Why v3?

`params-proto` v3 is a complete redesign focused on simplicity, that maximally takes
advantage of Python's new type hint system. This new API is in fact quite similar to
`params-proto` v1, but at that time when I developed v1 (back in 2018), the python
type hint system was still in its infancy. v2 was a rewrite from v1 that introduced
the `params_proto.hyper` module, that allowed us to create and load hyperparameter
sweeps.

v2 simply returns us to that, but with a modern, polished, and unified API for both
decorator-based cli programs and the same powerful parameter sweeps.

| Feature     | v1            | v2                | v3           |
|-------------|---------------|-------------------|--------------|
| API Style   | Decorators    | Class inheritance | Decorators   |
| Type Hints  | Not available | Optional          | Required     |
| Inline Docs | Manual        | Manual            | Automatic    |
| Functions   | Full support  | Not supported     | Full support |
| Union Types | Not supported | Limited           | Full support |
| IDE Support | Basic         | Basic             | Excellent    |

See the [Migration Guide](migration.md) for upgrading from v2.

## GitHub Repository

The source code is available on [GitHub](https://github.com/geyang/params-proto).

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
