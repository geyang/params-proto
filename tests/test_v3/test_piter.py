"""Tests for piter (ParameterIterator) functionality."""

import json

import pytest

from params_proto import Sweep, proto
from params_proto.hyper import ParameterIterator, piter


def test_piter_basic():
  """Test basic piter creation and iteration (zip by default)."""
  configs = piter @ {"lr": [0.001, 0.01], "batch_size": [32, 64]}

  # Should be lazy - not materialized yet
  assert isinstance(configs, ParameterIterator)

  # Materialize and check (zipped element-wise)
  config_list = configs.to_list()
  assert len(config_list) == 2  # Zipped: 2 elements
  assert config_list[0] == {"lr": 0.001, "batch_size": 32}
  assert config_list[1] == {"lr": 0.01, "batch_size": 64}


def test_piter_single_value():
  """Test piter with single values."""
  configs = piter @ {"seed": 200, "lr": 0.001}

  config_list = list(configs)
  assert len(config_list) == 1
  assert config_list[0] == {"seed": 200, "lr": 0.001}


def test_piter_product():
  """Test Cartesian product with * operator."""
  # New @ syntax with chained dicts
  combined = piter @ {"lr": [0.001, 0.01]} * {"batch_size": [32, 64]}

  config_list = list(combined)
  assert len(config_list) == 4
  assert config_list[0] == {"lr": 0.001, "batch_size": 32}
  assert config_list[1] == {"lr": 0.001, "batch_size": 64}
  assert config_list[2] == {"lr": 0.01, "batch_size": 32}
  assert config_list[3] == {"lr": 0.01, "batch_size": 64}


def test_piter_mod_dict():
  """Test override with % operator using dict."""
  configs = piter @ {"batch_size": [32, 64, 128]}
  with_lr = configs % {"lr": 0.001}

  config_list = list(with_lr)
  assert len(config_list) == 3
  assert config_list[0] == {"batch_size": 32, "lr": 0.001}
  assert config_list[1] == {"batch_size": 64, "lr": 0.001}
  assert config_list[2] == {"batch_size": 128, "lr": 0.001}


def test_piter_mod_piter():
  """Test override with % operator using another piter."""
  configs = piter @ {"batch_size": [32, 64, 128]}
  with_lr = configs % (piter @ {"lr": 0.001, "seed": 200})

  config_list = list(with_lr)
  assert len(config_list) == 3
  assert config_list[0] == {"batch_size": 32, "lr": 0.001, "seed": 200}
  assert config_list[1] == {"batch_size": 64, "lr": 0.001, "seed": 200}
  assert config_list[2] == {"batch_size": 128, "lr": 0.001, "seed": 200}


def test_piter_power():
  """Test repeat with ** operator."""
  configs = piter @ {"lr": [0.001, 0.01]}
  repeated = configs ** 3

  config_list = list(repeated)
  assert len(config_list) == 6  # 2 configs x 3 repetitions
  assert config_list[0] == {"lr": 0.001}
  assert config_list[1] == {"lr": 0.001}
  assert config_list[2] == {"lr": 0.001}
  assert config_list[3] == {"lr": 0.01}
  assert config_list[4] == {"lr": 0.01}
  assert config_list[5] == {"lr": 0.01}


def test_piter_combined_operations():
  """Test combining multiple piter operations."""
  # Product then override using chain syntax
  combined = (piter @ {"lr": [0.001, 0.01]} * {"batch_size": [32, 64]}) % {"seed": 42}

  config_list = list(combined)
  assert len(config_list) == 4
  assert all(c["seed"] == 42 for c in config_list)
  assert config_list[0] == {"lr": 0.001, "batch_size": 32, "seed": 42}


def test_piter_with_prefixes():
  """Test piter with prefixed parameter names (zipped)."""
  configs = piter @ {"model.depth": [18, 50], "training.lr": [0.001, 0.01]}

  config_list = list(configs)
  assert len(config_list) == 2  # Zipped
  assert config_list[0] == {"model.depth": 18, "training.lr": 0.001}
  assert config_list[1] == {"model.depth": 50, "training.lr": 0.01}


def test_piter_reusability():
  """Test that piter can be iterated multiple times."""
  configs = piter @ {"lr": [0.001, 0.01]}

  # First iteration
  list1 = list(configs)
  # Second iteration - should work since list is cached
  list2 = list(configs)

  assert list1 == list2
  assert len(list1) == 2


def test_piter_type_errors():
  """Test that piter raises appropriate type errors."""

  # Non-string keys (keep backward compat syntax for error testing)
  with pytest.raises(TypeError, match="must be strings"):
    piter({123: [1, 2, 3]})

  # Invalid * operand
  configs = piter @ {"lr": [0.001, 0.01]}
  with pytest.raises(TypeError, match="Cannot multiply"):
    configs * [1, 2, 3]

  # Invalid % operand
  with pytest.raises(TypeError, match="must be dict or ParameterIterator"):
    configs % [1, 2, 3]

  # Invalid ** operand
  with pytest.raises(ValueError, match="must be a positive integer"):
    configs ** -1

  with pytest.raises(ValueError, match="must be a positive integer"):
    configs ** 0


def test_piter_empty_override():
  """Test that empty piter raises error when used with %."""
  configs = piter @ {"lr": [0.001, 0.01]}
  empty = piter @ {}

  with pytest.raises(ValueError, match="empty"):
    configs % empty


def test_piter_len():
  """Test len() on ParameterIterator."""
  configs = piter @ {"lr": [0.001, 0.01], "batch_size": [32, 64]}

  # len() should materialize the list
  assert len(configs) == 2  # Zipped


def test_piter_empty_spec():
  """Test piter with empty dict."""
  configs = piter @ {}
  config_list = list(configs)
  assert len(config_list) == 0


def test_piter_range_values():
  """Test piter with range objects."""
  configs = piter @ {"seed": range(5), "lr": [0.001, 0.01, 0.1, 0.5, 1.0]}
  config_list = list(configs)

  assert len(config_list) == 5  # Zipped
  assert config_list[0] == {"seed": 0, "lr": 0.001}
  assert config_list[-1] == {"seed": 4, "lr": 1.0}


def test_piter_tuple_values():
  """Test piter with tuple values."""
  configs = piter @ {"lr": (0.001, 0.01), "batch_size": (32, 64)}
  config_list = list(configs)

  assert len(config_list) == 2  # Zipped
  assert config_list[0] == {"lr": 0.001, "batch_size": 32}
  assert config_list[1] == {"lr": 0.01, "batch_size": 64}


def test_piter_chained_operations():
  """Test chaining multiple piter operations."""
  # Use chain syntax: piter @ {...} * {...}
  result = ((piter @ {"lr": [0.001, 0.01]} * {"batch_size": [32, 64]}) % {"seed": 42}) ** 2

  config_list = list(result)
  assert len(config_list) == 8  # 2 lr × 2 batch × 2 repeats
  assert all(c["seed"] == 42 for c in config_list)


def test_piter_override_precedence():
  """Test that % overrides existing keys."""
  configs = piter @ {"lr": [0.001, 0.01], "seed": [1, 2]}
  overridden = configs % {"seed": 42, "new_param": "value"}

  config_list = list(overridden)
  assert len(config_list) == 2  # Zipped
  # All should have seed=42 (overridden)
  assert all(c["seed"] == 42 for c in config_list)
  assert all(c["new_param"] == "value" for c in config_list)


def test_piter_multiple_iterations():
  """Test that piter can be iterated multiple times with caching."""
  configs = piter @ {"lr": [0.001, 0.01]}

  # First iteration
  list1 = [c for c in configs]

  # Second iteration should use cached results
  list2 = [c for c in configs]

  # Third iteration
  list3 = list(configs)

  assert list1 == list2 == list3
  assert len(list1) == 2


def test_piter_nested_dicts_in_values():
  """Test that nested structures in values work correctly."""
  configs = piter @ {
      "optimizer": ["adam", "sgd"],
      "config": [{"momentum": 0.9}, {"momentum": 0.95}]
  }

  config_list = list(configs)
  assert len(config_list) == 2  # Zipped
  assert config_list[0] == {"optimizer": "adam", "config": {"momentum": 0.9}}
  assert config_list[1] == {"optimizer": "sgd", "config": {"momentum": 0.95}}


def test_piter_preserves_types():
  """Test that piter preserves value types."""
  configs = piter @ {
      "lr": [0.001, 0.01],  # float
      "batch_size": [32, 64],  # int
      "name": ["exp1", "exp2"],  # str
      "enabled": [True, False],  # bool
  }

  first_config = next(iter(configs))
  assert isinstance(first_config["lr"], float)
  assert isinstance(first_config["batch_size"], int)
  assert isinstance(first_config["name"], str)
  assert isinstance(first_config["enabled"], bool)


def test_piter_large_cartesian_product():
  """Test piter with large Cartesian product (memory efficiency)."""
  # This creates a large product space when using * operator with chain syntax
  configs = piter @ {"param1": range(10)} * {"param2": range(10)} * {"param3": range(10)}  # 1000 configs

  # Should be able to iterate first 5 without materializing all
  first_five = []
  for i, config in enumerate(configs):
    if i >= 5:
      break
    first_five.append(config)

  assert len(first_five) == 5
  assert first_five[0] == {"param1": 0, "param2": 0, "param3": 0}


def test_piter_complex_composition():
  """Test complex real-world composition pattern."""
  # Define base experiments using @ syntax
  models = piter @ {"model": ["resnet18", "vit"]}
  datasets = piter @ {"dataset": ["cifar10", "imagenet"]}

  # Hyperparameters - use product for grid search with chain syntax
  lr_schedule = piter @ {"lr": [0.001, 0.01]} * {"lr_decay": [0.1, 0.5]}

  # Compose
  experiments = (models * datasets * lr_schedule) % {"seed": 42}

  # Run 3 trials per config
  all_runs = experiments ** 3

  config_list = list(all_runs)
  assert len(config_list) == 48  # 2 models × 2 datasets × 2 lr × 2 decay × 3 trials
  assert all(c["seed"] == 42 for c in config_list)

  # Check first and last to ensure correctness
  assert config_list[0]["model"] == "resnet18"
  assert config_list[0]["dataset"] == "cifar10"


def test_piter_integration_with_sweep_save_load(tmp_path):
  """Test that piter results can be saved/loaded via Sweep."""
  @proto.prefix
  class Config:
    lr: float = 0.001
    batch_size: int = 32

  # Create piter configs using @ syntax
  piter_configs = piter @ {
      "config.lr": [0.001, 0.01],
      "config.batch_size": [32, 64]
  }

  # Convert to list and save manually
  configs_list = piter_configs.to_list()

  sweep_file = tmp_path / "piter_sweep.jsonl"
  with open(sweep_file, "w") as f:
    for config in configs_list:
      f.write(json.dumps(config) + "\n")

  # Load via Sweep
  sweep = Sweep(Config).load(str(sweep_file))
  loaded_configs = list(sweep)

  assert len(loaded_configs) == 2  # Zipped
  assert loaded_configs[0] == {"config.lr": 0.001, "config.batch_size": 32}


def test_piter_mod_with_nested_structure():
  """Test % operator with complex nested overrides."""
  configs = piter @ {"lr": [0.001, 0.01]}

  # Override with nested structure
  overridden = configs % {
      "seed": 42,
      "logging": {"level": "INFO", "dir": "/tmp/logs"},
      "device": "cuda"
  }

  config_list = list(overridden)
  assert len(config_list) == 2
  assert config_list[0]["logging"] == {"level": "INFO", "dir": "/tmp/logs"}
  assert config_list[0]["seed"] == 42


def test_piter_pow_with_zero():
  """Test ** operator with zero (should raise error)."""
  configs = piter @ {"lr": [0.001, 0.01]}

  with pytest.raises(ValueError, match="positive integer"):
    configs ** 0


def test_piter_pow_with_one():
  """Test ** operator with one (identity)."""
  configs = piter @ {"lr": [0.001, 0.01]}
  repeated = configs ** 1

  config_list = list(repeated)
  assert len(config_list) == 2
  assert config_list[0] == {"lr": 0.001}
  assert config_list[1] == {"lr": 0.01}


def test_piter_special_characters_in_keys():
  """Test piter with special characters in parameter names."""
  configs = piter @ {
      "model.conv.depth": [18, 50],
      "training.optimizer.lr": [0.001, 0.01],
      "data/path": ["/path1", "/path2"]
  }

  config_list = list(configs)
  assert len(config_list) == 2  # Zipped
  assert "model.conv.depth" in config_list[0]
  assert "data/path" in config_list[0]
  assert config_list[0] == {
      "model.conv.depth": 18,
      "training.optimizer.lr": 0.001,
      "data/path": "/path1"
  }


def test_piter_save(tmp_path):
  """Test saving ParameterIterator to JSONL file."""
  configs = piter @ {"lr": [0.001, 0.01]} * {"batch_size": [32, 64]}

  sweep_file = tmp_path / "test.jsonl"
  configs.save(sweep_file, verbose=False)

  # Verify file contents
  with open(sweep_file) as f:
    lines = f.readlines()

  assert len(lines) == 4  # 2 x 2 Cartesian product

  import json
  saved_configs = [json.loads(line) for line in lines]
  assert saved_configs[0] == {"lr": 0.001, "batch_size": 32}
  assert saved_configs[1] == {"lr": 0.001, "batch_size": 64}
  assert saved_configs[2] == {"lr": 0.01, "batch_size": 32}
  assert saved_configs[3] == {"lr": 0.01, "batch_size": 64}


def test_piter_load(tmp_path):
  """Test loading ParameterIterator from JSONL file."""
  # Create a JSONL file manually
  sweep_file = tmp_path / "test.jsonl"
  with open(sweep_file, "w") as f:
    f.write('{"lr": 0.001, "batch_size": 32}\n')
    f.write('{"lr": 0.01, "batch_size": 64}\n')

  # Load using ParameterIterator.load
  loaded = ParameterIterator.load(sweep_file)

  config_list = list(loaded)
  assert len(config_list) == 2
  assert config_list[0] == {"lr": 0.001, "batch_size": 32}
  assert config_list[1] == {"lr": 0.01, "batch_size": 64}


def test_piter_save_and_load_roundtrip(tmp_path):
  """Test save and load roundtrip preserves data."""
  # Create complex configs with operators
  configs = piter @ {"lr": [0.001, 0.01]} * {"batch_size": [32, 64]} % {"seed": 42}
  original_list = configs.to_list()

  # Save
  sweep_file = tmp_path / "roundtrip.jsonl"
  configs.save(sweep_file, verbose=False)

  # Load
  loaded = ParameterIterator.load(sweep_file)
  loaded_list = loaded.to_list()

  # Compare
  assert len(loaded_list) == len(original_list)
  for orig, loaded_item in zip(original_list, loaded_list):
    assert orig == loaded_item


def test_piter_save_append_mode(tmp_path):
  """Test save with append mode."""
  sweep_file = tmp_path / "append.jsonl"

  # First save
  configs1 = piter @ {"lr": [0.001, 0.01]}
  configs1.save(sweep_file, overwrite=True, verbose=False)

  # Append
  configs2 = piter @ {"lr": [0.1, 1.0]}
  configs2.save(sweep_file, overwrite=False, verbose=False)

  # Load all
  loaded = ParameterIterator.load(sweep_file)
  config_list = list(loaded)

  assert len(config_list) == 4
  assert config_list[0] == {"lr": 0.001}
  assert config_list[1] == {"lr": 0.01}
  assert config_list[2] == {"lr": 0.1}
  assert config_list[3] == {"lr": 1.0}


def test_piter_load_with_comments(tmp_path):
  """Test that load ignores comment lines (lines starting with //)."""
  sweep_file = tmp_path / "with_comments.jsonl"
  with open(sweep_file, "w") as f:
    f.write('// This is a comment\n')
    f.write('{"lr": 0.001}\n')
    f.write('// Another comment\n')
    f.write('{"lr": 0.01}\n')

  loaded = ParameterIterator.load(sweep_file)
  config_list = list(loaded)

  assert len(config_list) == 2
  assert config_list[0] == {"lr": 0.001}
  assert config_list[1] == {"lr": 0.01}


def test_piter_load_operations_on_loaded(tmp_path):
  """Test that loaded ParameterIterator supports all operations."""
  sweep_file = tmp_path / "base.jsonl"
  with open(sweep_file, "w") as f:
    f.write('{"lr": 0.001}\n')
    f.write('{"lr": 0.01}\n')

  loaded = ParameterIterator.load(sweep_file)

  # Test product
  product_result = loaded * {"batch_size": [32, 64]}
  assert len(list(product_result)) == 4

  # Test override
  override_result = loaded % {"seed": 42}
  config_list = list(override_result)
  assert all(c["seed"] == 42 for c in config_list)

  # Test repeat
  repeat_result = loaded ** 3
  assert len(list(repeat_result)) == 6
