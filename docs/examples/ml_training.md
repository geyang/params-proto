# ML Training Example

This example shows a complete machine learning training script using params-proto v3.

## Simple MNIST Training

Based on the test case `test_proto_cli.py::test_proto_cli`:

```python
from params_proto import proto

@proto.cli
def train_mnist(
    batch_size: int = 128,  # Training batch size
    epochs: int = 10,  # Number of training epochs
    lr: float = 0.001,  # Learning rate
    seed: int = 42,  # Random seed
):
    """Train an MLP on MNIST dataset."""
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms

    # Set random seed
    torch.manual_seed(seed)

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Define simple MLP
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    print("Training complete!")

if __name__ == "__main__":
    train_mnist()
```

## Usage

Run with defaults:
```bash
python train_mnist.py
```

Override parameters:
```bash
python train_mnist.py --lr 0.01 --batch-size 256 --epochs 20
```

Get help:
```bash
python train_mnist.py --help
```

Output:
```{ansi-block}
:string_escape:

usage: mnist_train.py [-h] [--batch-size \x1b[1m\x1b[94mINT\x1b[0m] [--epochs \x1b[1m\x1b[94mINT\x1b[0m] [--lr \x1b[1m\x1b[94mFLOAT\x1b[0m] [--seed \x1b[1m\x1b[94mINT\x1b[0m]

Train an MLP on MNIST dataset.

options:
  -h, --help           show this help message and exit
  --batch-size \x1b[1m\x1b[94mINT\x1b[0m     Training batch size \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m128\x1b[0m\x1b[36m)\x1b[0m
  --epochs \x1b[1m\x1b[94mINT\x1b[0m         Number of training epochs \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m10\x1b[0m\x1b[36m)\x1b[0m
  --lr \x1b[1m\x1b[94mFLOAT\x1b[0m           Learning rate \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m0.001\x1b[0m\x1b[36m)\x1b[0m
  --seed \x1b[1m\x1b[94mINT\x1b[0m           Random seed \x1b[36m(default:\x1b[0m \x1b[1m\x1b[36m42\x1b[0m\x1b[36m)\x1b[0m
```

## Advanced: Parameter Sweeps

Run multiple experiments with different hyperparameters:

```python
from params_proto import proto

# Override in code for sweeps
for lr in [0.001, 0.01, 0.1]:
    for batch_size in [32, 64, 128]:
        print(f"\n=== Training with lr={lr}, batch_size={batch_size} ===")
        with proto.bind(**{
            "train_mnist.lr": lr,
            "train_mnist.batch_size": batch_size,
        }):
            train_mnist(epochs=5)  # Quick test
```

## Next Steps

- See [RL Agent Example](rl_agent.md) for a more complex multi-config example
- Check [CLI Applications](cli_applications.md) for building command-line tools
