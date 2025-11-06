#!/usr/bin/env python
"""Example script demonstrating positional argument support in params-proto v3."""

from params_proto import proto


@proto.cli
def train(
  seed: int,  # Random seed (required)
  lr: float = 0.001,  # Learning rate
  batch_size: int = 32,  # Batch size
  epochs: int = 10,  # Number of epochs
):
  """Train a simple model with colorized CLI help.

  This example demonstrates that required parameters (like seed) can be passed
  either as positional arguments or as named arguments:

  Examples:
    python example_positional_cli.py 42
    python example_positional_cli.py 42 --lr 0.01
    python example_positional_cli.py --seed 42 --lr 0.01
  """
  print("Training with:")
  print(f"  seed={seed}")
  print(f"  lr={lr}")
  print(f"  batch_size={batch_size}")
  print(f"  epochs={epochs}")


if __name__ == "__main__":
  train()
