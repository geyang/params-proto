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
- Read [Prefixes Guide](../key_concepts/prefixes.md) for more on `@proto.prefix`
