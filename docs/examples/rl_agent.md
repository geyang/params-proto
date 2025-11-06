# RL Agent Example

This example shows a reinforcement learning agent configuration with multiple config groups.

Based on the test case `test_proto_cli.py::test_rl_agent_help`:

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
    import numpy as np

    print(f"Training {Agent.algorithm} on {Environment.domain}-{Environment.task}")
    print(f"Environment settings:")
    print(f"  Domain: {Environment.domain}")
    print(f"  Task: {Environment.task}")
    print(f"  Time limit: {Environment.time_limit}s")
    print(f"\nAgent settings:")
    print(f"  Algorithm: {Agent.algorithm}")
    print(f"  Buffer size: {Agent.buffer_size}")
    print(f"  Gamma: {Agent.gamma}")
    print(f"  Tau: {Agent.tau}")
    print(f"\nTraining settings:")
    print(f"  Total steps: {total_steps}")
    print(f"  Eval frequency: {eval_freq}")
    print(f"  Seed: {seed}")

    # Set random seed
    np.random.seed(seed)

    # Training loop (simplified)
    for step in range(0, total_steps, eval_freq):
        print(f"\nStep {step}/{total_steps}")
        # ... training code here ...

        if step % eval_freq == 0:
            print(f"  Evaluating at step {step}")
            # ... evaluation code here ...

    print("\nTraining complete!")

if __name__ == "__main__":
    train_rl()
```

## CLI Help Output

Running `python train_rl.py --help` shows:

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

## Usage Examples

Run with defaults:
```bash
python train_rl.py
```

Change environment:
```bash
python train_rl.py --Environment.domain walker --Environment.task walk
```

Adjust agent hyperparameters:
```bash
python train_rl.py --Agent.gamma 0.95 --Agent.tau 0.01
```

Full customization:
```bash
python train_rl.py \
  --seed 123 \
  --total-steps 500000 \
  --Environment.domain walker \
  --Environment.task walk \
  --Agent.algorithm PPO \
  --Agent.buffer-size 500000
```

## Key Features Demonstrated

1. **Modular Configuration**: Separate config groups for Environment and Agent
2. **Singleton Pattern**: `@proto.prefix` creates global singletons
3. **Automatic Grouping**: CLI help automatically groups options by prefix
4. **Inline Documentation**: Comments become help text
5. **Type Safety**: Full type annotations for all parameters

## Next Steps

- See [Basic Usage](basic_usage.md) for simpler examples
- Check [ML Training](ml_training.md) for another complete example
- Read [Prefixes Guide](../guide/prefixes.md) for more on `@proto.prefix`
