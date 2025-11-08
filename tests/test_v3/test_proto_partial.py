"""Tests for proto.partial() decorator in v3 API."""

import pytest
from params_proto import proto


def test_partial_basic():
  """Test basic partial decoration with plain class."""

  class Config:
    lr: float = 0.01
    batch_size: int = 32
    optimizer: str = "adam"

  @proto.partial(Config)
  def train(lr, batch_size, optimizer):
    return {"lr": lr, "batch_size": batch_size, "optimizer": optimizer}

  # Should use defaults from Config
  result = train()
  assert result["lr"] == 0.01
  assert result["batch_size"] == 32
  assert result["optimizer"] == "adam"


def test_partial_attribute_modification():
  """Test that modifying config attributes affects function calls."""

  class Config:
    lr: float = 0.01
    batch_size: int = 32

  @proto.partial(Config)
  def train(lr, batch_size):
    return {"lr": lr, "batch_size": batch_size}

  # Modify config
  Config.lr = 0.001
  Config.batch_size = 64

  result = train()
  assert result["lr"] == 0.001
  assert result["batch_size"] == 64


def test_partial_hyperparameter_sweep():
  """Test hyperparameter sweep pattern."""

  class Config:
    lr: float = 0.01
    batch_size: int = 32

  @proto.partial(Config)
  def train(lr, batch_size):
    return {"lr": lr, "batch_size": batch_size}

  # Test sweep
  results = []
  for Config.lr in [0.01, 0.001, 0.0001]:
    results.append(train())

  assert results[0]["lr"] == 0.01
  assert results[1]["lr"] == 0.001
  assert results[2]["lr"] == 0.0001
  assert all(r["batch_size"] == 32 for r in results)


def test_partial_with_function_defaults():
  """Test that function defaults take precedence over missing config values."""

  class Config:
    lr: float = 0.01
    # batch_size not in config

  @proto.partial(Config)
  def train(lr, batch_size=64, optimizer="sgd"):
    return {"lr": lr, "batch_size": batch_size, "optimizer": optimizer}

  result = train()
  assert result["lr"] == 0.01
  assert result["batch_size"] == 64  # Uses function default
  assert result["optimizer"] == "sgd"  # Uses function default


def test_partial_override_with_kwargs():
  """Test that explicit kwargs override config values."""

  class Config:
    lr: float = 0.01
    batch_size: int = 32

  @proto.partial(Config)
  def train(lr, batch_size):
    return {"lr": lr, "batch_size": batch_size}

  result = train(lr=0.1, batch_size=128)
  assert result["lr"] == 0.1
  assert result["batch_size"] == 128


def test_partial_with_keyword_only_args():
  """Test partial with keyword-only arguments."""

  class Config:
    lr: float = 0.01
    batch_size: int = 32
    optimizer: str = "adam"

  @proto.partial(Config)
  def train(lr, *, batch_size, optimizer):
    return {"lr": lr, "batch_size": batch_size, "optimizer": optimizer}

  # lr is positional, should not be filled by config when there are keyword-only without defaults
  with pytest.raises(TypeError):
    train()  # Missing required positional argument 'lr'

  # Providing lr should work
  result = train(0.001)
  assert result["lr"] == 0.001
  assert result["batch_size"] == 32
  assert result["optimizer"] == "adam"


def test_partial_mixed_parameters():
  """Test partial with mix of config and non-config parameters."""

  class Config:
    lr: float = 0.01
    batch_size: int = 32

  @proto.partial(Config)
  def train(lr, batch_size, epochs, model_name="resnet"):
    return {
      "lr": lr,
      "batch_size": batch_size,
      "epochs": epochs,
      "model_name": model_name
    }

  # epochs must be provided, lr and batch_size come from config
  result = train(epochs=100)
  assert result["lr"] == 0.01
  assert result["batch_size"] == 32
  assert result["epochs"] == 100
  assert result["model_name"] == "resnet"


def test_partial_no_function_params():
  """Test partial where function accesses config directly without parameters."""

  class Config:
    lr: float = 0.01
    batch_size: int = 32

  @proto.partial(Config)
  def train():
    # Access config directly
    return {"lr": Config.lr, "batch_size": Config.batch_size}

  result = train()
  assert result["lr"] == 0.01
  assert result["batch_size"] == 32

  # Modify and test again
  Config.lr = 0.001
  result = train()
  assert result["lr"] == 0.001


def test_partial_with_class_method():
  """Test partial decorator on class methods."""

  class Config:
    lr: float = 0.01
    batch_size: int = 32

  class Trainer:
    @proto.partial(Config, method=True)
    def train(self, lr, batch_size):
      return {"lr": lr, "batch_size": batch_size, "trainer": self}

  trainer = Trainer()
  result = trainer.train()
  assert result["lr"] == 0.01
  assert result["batch_size"] == 32
  assert result["trainer"] is trainer


def test_partial_preserves_function_metadata():
  """Test that partial preserves function name and docstring."""

  class Config:
    lr: float = 0.01

  @proto.partial(Config)
  def train(lr):
    """Train a model with the given learning rate."""
    return lr

  assert train.__name__ == "train"
  assert train.__doc__ == "Train a model with the given learning rate."


def test_partial_empty_config():
  """Test partial with empty config class."""

  class Config:
    pass

  @proto.partial(Config)
  def train(lr=0.01, batch_size=32):
    return {"lr": lr, "batch_size": batch_size}

  # Should use function defaults
  result = train()
  assert result["lr"] == 0.01
  assert result["batch_size"] == 32


def test_partial_with_positional_args():
  """Test that positional args work correctly."""

  class Config:
    lr: float = 0.01
    batch_size: int = 32

  @proto.partial(Config)
  def train(lr, batch_size, epochs):
    return {"lr": lr, "batch_size": batch_size, "epochs": epochs}

  # Pass epochs as keyword argument (since lr and batch_size come from config)
  result = train(epochs=100)
  assert result["lr"] == 0.01
  assert result["batch_size"] == 32
  assert result["epochs"] == 100

  # Override lr and batch_size with positional args
  result = train(0.001, 64, 200)
  assert result["lr"] == 0.001
  assert result["batch_size"] == 64
  assert result["epochs"] == 200

  # Override just lr with positional, rest from config/kwargs
  result = train(0.002, epochs=150)
  assert result["lr"] == 0.002
  assert result["batch_size"] == 32  # from config
  assert result["epochs"] == 150
