#!/usr/bin/env python
"""Simple demo of params-proto v3 with ANSI colored help."""

from params_proto import proto


@proto.cli
def train(
  seed: int,  # Random seed
  lr: float = 0.001,  # Learning rate
  batch_size: int = 32,  # Batch size
  epochs: int = 10,  # Number of epochs
):
  """Train a simple model with colorized CLI help."""
  print("Training with:")
  print(f"  seed={seed}")
  print(f"  lr={lr}")
  print(f"  batch_size={batch_size}")
  print(f"  epochs={epochs}")


if __name__ == "__main__":
  train()
