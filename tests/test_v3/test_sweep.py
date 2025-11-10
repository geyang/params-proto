"""Tests for Sweep with v3 @proto decorator API."""

import pytest

from params_proto import Sweep, proto
from params_proto.hyper import ParameterIterator, piter


def test_product():
  """Test Cartesian product of parameter lists."""

  @proto
  class Config:
    start_seed: int = 10
    discern_flag: bool = True
    env_name: str = "dm_lab"
    batch_size: int = 5

  with Sweep(Config).product as sweep:
    Config.start_seed = range(2)
    Config.discern_flag = [True, False]

  all_configs = list(sweep)
  assert all_configs == [
    {"discern_flag": True, "start_seed": 0},
    {"discern_flag": False, "start_seed": 0},
    {"discern_flag": True, "start_seed": 1},
    {"discern_flag": False, "start_seed": 1},
  ]


def test_zip():
  """Test element-wise zipping of parameter lists."""

  @proto
  class Config:
    start_seed: int = 10
    discern_flag: bool = True
    env_name: str = "dm_lab"
    batch_size: int = 5

  with Sweep(Config).zip as sweep:
    Config.env_name = ["small_env", "large_env"]
    Config.batch_size = [10, 50]

  all_configs = list(sweep)
  assert all_configs == [
    {"batch_size": 10, "env_name": "small_env"},
    {"batch_size": 50, "env_name": "large_env"},
  ]


def test_multiple_configs_with_prefix():
  """Test sweeping over multiple @proto.prefix classes."""

  @proto.prefix
  class Model:
    a: int = 5

  @proto.prefix
  class Training:
    b: int = 10

  with Sweep(Model, Training).zip as sweep:
    Model.a = range(100)
    Training.b = range(100)

  for i, all_deps in enumerate(sweep):
    assert Model.a == i
    assert Training.b == i

  # Test combining set and product
  with Sweep(Model, Training) as sweep:
    with sweep.product:
      Training.b = range(100)

  for i, deps in enumerate(sweep):
    assert deps == {"training.b": i}
    assert Training.b == i

  # Test that setting non-existent attributes raises error
  with pytest.raises(AttributeError, match="Cannot set non-existent attribute"):
    with Sweep(Model, Training) as sweep:
      with sweep.set:
        Training.nonexist_gets_written = "hey"


def test_product_zip_nested():
  """Test nested product and zip operators."""

  @proto.prefix
  class Config:
    start_seed: int = 10
    discern_flag: bool = True
    env_name: str = "dm_lab"
    batch_size: int = 5

  with Sweep(Config) as sweep:
    with sweep.product:
      Config.start_seed = range(2)
      Config.discern_flag = [True, False]

      with sweep.zip:
        Config.env_name = ["small_env", "large_env"]
        Config.batch_size = [10, 50]

  all_configs = list(sweep)
  assert len(all_configs) == 2 * 2 * 2
  assert all_configs == [
    {
      "config.batch_size": 10,
      "config.discern_flag": True,
      "config.env_name": "small_env",
      "config.start_seed": 0,
    },
    {
      "config.batch_size": 50,
      "config.discern_flag": True,
      "config.env_name": "large_env",
      "config.start_seed": 0,
    },
    {
      "config.batch_size": 10,
      "config.discern_flag": False,
      "config.env_name": "small_env",
      "config.start_seed": 0,
    },
    {
      "config.batch_size": 50,
      "config.discern_flag": False,
      "config.env_name": "large_env",
      "config.start_seed": 0,
    },
    {
      "config.batch_size": 10,
      "config.discern_flag": True,
      "config.env_name": "small_env",
      "config.start_seed": 1,
    },
    {
      "config.batch_size": 50,
      "config.discern_flag": True,
      "config.env_name": "large_env",
      "config.start_seed": 1,
    },
    {
      "config.batch_size": 10,
      "config.discern_flag": False,
      "config.env_name": "small_env",
      "config.start_seed": 1,
    },
    {
      "config.batch_size": 50,
      "config.discern_flag": False,
      "config.env_name": "large_env",
      "config.start_seed": 1,
    },
  ]


def test_set():
  """Test setting fixed values for all configs."""

  @proto.prefix
  class Config:
    start_seed: int = 10
    discern_flag: bool = True
    env_name: str = "dm_lab"
    batch_size: int = 5

  with Sweep(Config) as sweep:
    Config.start_seed = 20
    Config.discern_flag = False

  for override in sweep:
    assert Config.discern_flag is False
    assert override == {"config.discern_flag": False, "config.start_seed": 20}


def test_set_with_zip():
  """Test combining set and zip operators."""

  @proto
  class Config:
    start_seed: int = 10
    discern_flag: bool = True
    env_name: str = "dm_lab"
    batch_size: int = 5
    replicas_hint: list = ["yo", "hey"]

  with Sweep(Config) as sweep:
    Config.replicas_hint = 12

    with sweep.zip:
      Config.env_name = ["small", "large"]
      Config.batch_size = [10, 50]

  all_configs = list(sweep)
  assert len(all_configs) == 2


def test_chain():
  """Test chaining multiple sweep configurations."""

  @proto
  class Config:
    level_name: str = "dmlab"
    some_prefix: str = "dmlab/1"
    start_seed: int = 10

  with Sweep(Config) as sweep:
    Config.level_name = "gotham"
    Config.some_prefix = "gotham/1"
    with sweep.product:
      Config.start_seed = range(15)

  with sweep.set:
    Config.level_name = "dmlab"
    Config.some_prefix = "dmlab/1"
    with sweep.product:
      Config.start_seed = range(15)

  all_configs = list(sweep)
  assert len(all_configs) == 30


def test_chain_with_shared_root_set():
  """Test chain operator with shared root settings."""

  @proto
  class Config:
    root_set: bool = False
    level_name: str = "dmlab"
    some_prefix: str = "dmlab/1"
    start_seed: int = 10

  with Sweep(Config) as sweep:
    Config.root_set = True

    with sweep.chain:
      with sweep.set:
        Config.level_name = "gotham"
        Config.some_prefix = "gotham/1"

        with sweep.product:
          Config.start_seed = range(15)

      with sweep.set:
        Config.level_name = "dmlab"
        Config.some_prefix = "dmlab/1"

        with sweep.product:
          Config.start_seed = range(15)

  all_configs = list(sweep)
  assert len(all_configs) == 30


def test_jagged():
  """Test jagged configs with different keys."""

  @proto
  class Config:
    config_1: bool = False
    config_2: bool = False

  with Sweep(Config) as sweep:
    with sweep.chain:
      with sweep.set:
        Config.config_1 = 10
      with sweep.set:
        Config.config_2 = 20

  for i, deps in enumerate(sweep):
    if i == 0:
      assert Config.config_1 == 10
      assert Config.config_2 is False
    if i == 1:
      assert Config.config_1 is False
      assert Config.config_2 == 20


def test_subscription():
  """Test indexing and slicing sweep configs."""

  @proto
  class Config:
    start_seed: int = 10

  with Sweep(Config).product as sweep:
    Config.start_seed = list(range(100))

  # Test various slicing operations
  assert list(sweep[:5]) == [{"start_seed": i} for i in range(5)]
  assert list(sweep[10:20:3]) == [{"start_seed": i} for i in range(10, 20, 3)]
  assert list(sweep[30]) == [{"start_seed": 30}]
  assert list(sweep[1:]) == [{"start_seed": i} for i in range(1, 100)]


def test_negative_subscription():
  """Test negative indexing."""

  @proto
  class Config:
    start_seed: int = 10

  with Sweep(Config).product as sweep:
    Config.start_seed = list(range(100))

  result = list(sweep[-10:-5])
  expected = [{"start_seed": i} for i in range(90, 95)]
  for a, b in zip(result, expected, strict=False):
    assert a == b


def test_each():
  """Test .each() for computing derived parameters."""

  @proto.prefix
  class Config:
    seed: int = 10
    postfix: str = "seed-"

  with Sweep(Config).product as sweep:
    Config.seed = [10, 20, 30]

  with Sweep(Config):
    Config.seed = 110

  @sweep.each
  def each(Config):
    Config.postfix = f"config.seed-({Config.seed})"

  all_configs = list(sweep)

  assert all_configs[0]["config.postfix"] == "config.seed-(10)"
  assert all_configs[1]["config.postfix"] == "config.seed-(20)"
  assert all_configs[2]["config.postfix"] == "config.seed-(30)"


def test_set_getter():
  """Test getting values immediately after setting in Sweep context."""

  @proto.prefix
  class Config:
    seed: int = 10
    postfix: str = "seed-"

  with Sweep(Config).product as sweep:
    Config.seed = [10, 20, 30]

  with Sweep(Config):
    Config.seed = 110
    assert Config.seed == 110, "should be able to use the value"
    assert Config.postfix == "seed-", "should be able to get original value"

  @sweep.each
  def each(Config):
    Config.postfix = f"config.seed-({Config.seed})"
    assert Config.postfix == f"config.seed-({Config.seed})", (
      "should be able to get the updated value right away"
    )

  all_configs = list(sweep)

  assert all_configs[0]["config.postfix"] == "config.seed-(10)"
  assert all_configs[1]["config.postfix"] == "config.seed-(20)"
  assert all_configs[2]["config.postfix"] == "config.seed-(30)"


def test_save_and_load(tmp_path):
  """Test saving and loading sweep configurations."""

  @proto.prefix
  class Config:
    lr: float = 0.001
    batch_size: int = 32

  # Create a sweep
  with Sweep(Config).product as sweep:
    Config.lr = [0.001, 0.01, 0.1]
    Config.batch_size = [32, 64]

  # Save to file
  sweep_file = tmp_path / "test_sweep.jsonl"
  sweep.save(sweep_file, verbose=False)

  # Load from file
  loaded_sweep = Sweep(Config).load(sweep_file)
  loaded_configs = list(loaded_sweep)

  # Verify
  assert len(loaded_configs) == 6
  assert loaded_configs[0] == {"config.lr": 0.001, "config.batch_size": 32}


def test_dataframe():
  """Test converting sweep to pandas DataFrame."""

  @proto.prefix
  class Config:
    lr: float = 0.001
    batch_size: int = 32

  with Sweep(Config).product as sweep:
    Config.lr = [0.001, 0.01]
    Config.batch_size = [32, 64]

  df = sweep.dataframe
  assert len(df) == 4
  assert list(df.columns) == ["config.lr", "config.batch_size"]


def test_list_property():
  """Test sweep.list property."""

  @proto
  class Config:
    seed: int = 10

  with Sweep(Config).product as sweep:
    Config.seed = [1, 2, 3]

  configs_list = sweep.list
  assert configs_list == [
    {"seed": 1},
    {"seed": 2},
    {"seed": 3},
  ]


def test_load_with_strict_mode():
  """Test load with strict=False for missing attributes."""

  @proto.prefix
  class Config:
    config_1: bool = False

  @proto
  class Other:
    config_2: bool = False

  # Should work with valid keys
  sweep = Sweep(Config, Other).load([{"config.config_1": True}])

  # Should work with strict=False for missing keys
  sweep = Sweep(Config, Other).load([{"config.config_2": True}], strict=False)
  sweep = Sweep(Config, Other).load([{"does_not_exist": True}], strict=False)

  # Should raise KeyError with strict=True (default)
  with pytest.raises(KeyError):
    sweep = Sweep(Config, Other).load([{"config.config_2": True}])

  with pytest.raises(KeyError):
    sweep = Sweep(Config, Other).load([{"does_not_exist": True}])


def test_sweep_with_cli_function():
  """Test Sweep with @proto.cli decorated function."""

  @proto.prefix
  class Params:
    lr: float = 0.001
    batch_size: int = 32

  @proto.cli
  def train(seed: int):
    """Train function."""
    return seed

  # Create sweep with both class and function
  sweep = Sweep(Params, train)

  with sweep.zip:
    train.seed = [100, 100, 100]
    Params.batch_size = [32, 64, 128]
    Params.lr = [0.001, 0.01, 0.1]

  configs = tuple(sweep)
  assert len(configs) == 3
  assert configs[0] == {"seed": 100, "params.batch_size": 32, "params.lr": 0.001}
  assert configs[1] == {"seed": 100, "params.batch_size": 64, "params.lr": 0.01}
  assert configs[2] == {"seed": 100, "params.batch_size": 128, "params.lr": 0.1}


# ============================================================================
# piter (ParameterIterator) Tests
# ============================================================================


def test_piter_basic():
  """Test basic piter creation and iteration."""
  configs = piter({"lr": [0.001, 0.01], "batch_size": [32, 64]})

  # Should be lazy - not materialized yet
  assert isinstance(configs, ParameterIterator)

  # Materialize and check
  config_list = configs.to_list()
  assert len(config_list) == 4  # 2 x 2 = 4
  assert config_list[0] == {"lr": 0.001, "batch_size": 32}
  assert config_list[1] == {"lr": 0.001, "batch_size": 64}
  assert config_list[2] == {"lr": 0.01, "batch_size": 32}
  assert config_list[3] == {"lr": 0.01, "batch_size": 64}


def test_piter_single_value():
  """Test piter with single values."""
  configs = piter({"seed": 200, "lr": 0.001})

  config_list = list(configs)
  assert len(config_list) == 1
  assert config_list[0] == {"seed": 200, "lr": 0.001}


def test_piter_product():
  """Test Cartesian product with * operator."""
  piter1 = piter({"lr": [0.001, 0.01]})
  piter2 = piter({"batch_size": [32, 64]})

  combined = piter1 * piter2

  config_list = list(combined)
  assert len(config_list) == 4
  assert config_list[0] == {"lr": 0.001, "batch_size": 32}
  assert config_list[1] == {"lr": 0.001, "batch_size": 64}
  assert config_list[2] == {"lr": 0.01, "batch_size": 32}
  assert config_list[3] == {"lr": 0.01, "batch_size": 64}


def test_piter_mod_dict():
  """Test override with % operator using dict."""
  configs = piter({"batch_size": [32, 64, 128]})
  with_lr = configs % {"lr": 0.001}

  config_list = list(with_lr)
  assert len(config_list) == 3
  assert config_list[0] == {"batch_size": 32, "lr": 0.001}
  assert config_list[1] == {"batch_size": 64, "lr": 0.001}
  assert config_list[2] == {"batch_size": 128, "lr": 0.001}


def test_piter_mod_piter():
  """Test override with % operator using another piter."""
  configs = piter({"batch_size": [32, 64, 128]})
  with_lr = configs % piter({"lr": 0.001, "seed": 200})

  config_list = list(with_lr)
  assert len(config_list) == 3
  assert config_list[0] == {"batch_size": 32, "lr": 0.001, "seed": 200}
  assert config_list[1] == {"batch_size": 64, "lr": 0.001, "seed": 200}
  assert config_list[2] == {"batch_size": 128, "lr": 0.001, "seed": 200}


def test_piter_power():
  """Test repeat with ** operator."""
  configs = piter({"lr": [0.001, 0.01]})
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
  # Create base sweep
  piter1 = piter({"lr": [0.001, 0.01]})
  piter2 = piter({"batch_size": [32, 64]})

  # Product then override
  combined = (piter1 * piter2) % {"seed": 42}

  config_list = list(combined)
  assert len(config_list) == 4
  assert all(c["seed"] == 42 for c in config_list)
  assert config_list[0] == {"lr": 0.001, "batch_size": 32, "seed": 42}


def test_piter_with_prefixes():
  """Test piter with prefixed parameter names."""
  configs = piter({"model.depth": [18, 50], "training.lr": [0.001, 0.01]})

  config_list = list(configs)
  assert len(config_list) == 4
  assert config_list[0] == {"model.depth": 18, "training.lr": 0.001}
  assert config_list[1] == {"model.depth": 18, "training.lr": 0.01}


def test_piter_reusability():
  """Test that piter can be iterated multiple times."""
  configs = piter({"lr": [0.001, 0.01]})

  # First iteration
  list1 = list(configs)
  # Second iteration - should work since list is cached
  list2 = list(configs)

  assert list1 == list2
  assert len(list1) == 2


def test_sweep_mul_operator():
  """Test Sweep * Sweep operator."""

  @proto
  class Config1:
    lr: float = 0.001

  @proto
  class Config2:
    batch_size: int = 32

  sweep1 = Sweep(Config1)
  with sweep1.product:
    Config1.lr = [0.001, 0.01]

  sweep2 = Sweep(Config2)
  with sweep2.product:
    Config2.batch_size = [32, 64]

  combined = sweep1 * sweep2

  config_list = list(combined)
  assert len(config_list) == 4
  assert config_list[0] == {"lr": 0.001, "batch_size": 32}


def test_sweep_mod_operator_with_dict():
  """Test Sweep % dict operator."""

  @proto
  class Config:
    batch_size: int = 32

  sweep = Sweep(Config)
  with sweep.product:
    Config.batch_size = [32, 64, 128]

  with_lr = sweep % {"lr": 0.001}

  config_list = list(with_lr)
  assert len(config_list) == 3
  assert all(c["lr"] == 0.001 for c in config_list)


def test_sweep_mod_operator_with_piter():
  """Test Sweep % piter operator."""

  @proto
  class Config:
    batch_size: int = 32

  sweep = Sweep(Config)
  with sweep.product:
    Config.batch_size = [32, 64, 128]

  with_lr = sweep % piter({"lr": 0.001, "seed": 200})

  config_list = list(with_lr)
  assert len(config_list) == 3
  assert all(c["lr"] == 0.001 and c["seed"] == 200 for c in config_list)


def test_sweep_pow_operator():
  """Test Sweep ** n operator."""

  @proto
  class Config:
    lr: float = 0.001

  sweep = Sweep(Config)
  with sweep.product:
    Config.lr = [0.001, 0.01]

  repeated = sweep ** 3

  config_list = list(repeated)
  assert len(config_list) == 6  # 2 configs x 3 repetitions
  assert config_list[0] == {"lr": 0.001}
  assert config_list[1] == {"lr": 0.001}
  assert config_list[2] == {"lr": 0.001}


def test_piter_type_errors():
  """Test that piter raises appropriate type errors."""

  # Non-string keys
  with pytest.raises(TypeError, match="must be strings"):
    piter({123: [1, 2, 3]})

  # Invalid * operand
  configs = piter({"lr": [0.001, 0.01]})
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
  configs = piter({"lr": [0.001, 0.01]})
  empty = piter({})

  with pytest.raises(ValueError, match="empty"):
    configs % empty


def test_piter_len():
  """Test len() on ParameterIterator."""
  configs = piter({"lr": [0.001, 0.01], "batch_size": [32, 64]})

  # len() should materialize the list
  assert len(configs) == 4


def test_sweep_to_list():
  """Test Sweep.to_list() method."""

  @proto
  class Config:
    lr: float = 0.001

  sweep = Sweep(Config)
  with sweep.product:
    Config.lr = [0.001, 0.01, 0.1]

  list1 = sweep.to_list()
  list2 = sweep.list

  assert list1 == list2
  assert len(list1) == 3


# ============================================================================
# Additional piter Edge Cases and Integration Tests
# ============================================================================


def test_piter_empty_spec():
  """Test piter with empty dict."""
  configs = piter({})
  config_list = list(configs)
  assert len(config_list) == 0


def test_piter_range_values():
  """Test piter with range objects."""
  configs = piter({"seed": range(5), "lr": [0.001, 0.01]})
  config_list = list(configs)
  
  assert len(config_list) == 10  # 5 × 2
  assert config_list[0] == {"seed": 0, "lr": 0.001}
  assert config_list[-1] == {"seed": 4, "lr": 0.01}


def test_piter_tuple_values():
  """Test piter with tuple values."""
  configs = piter({"lr": (0.001, 0.01), "batch_size": (32, 64)})
  config_list = list(configs)
  
  assert len(config_list) == 4
  assert config_list[0] == {"lr": 0.001, "batch_size": 32}


def test_piter_chained_operations():
  """Test chaining multiple piter operations."""
  result = ((
      piter({"lr": [0.001, 0.01]}) *
      piter({"batch_size": [32, 64]})
  ) % {"seed": 42}) ** 2

  config_list = list(result)
  assert len(config_list) == 8  # 2 lr × 2 batch × 2 repeats
  assert all(c["seed"] == 42 for c in config_list)


def test_piter_override_precedence():
  """Test that % overrides existing keys."""
  configs = piter({"lr": [0.001, 0.01], "seed": [1, 2]})
  overridden = configs % {"seed": 42, "new_param": "value"}
  
  config_list = list(overridden)
  assert len(config_list) == 4  # 2 lr × 2 seed (before override)
  # All should have seed=42 (overridden)
  assert all(c["seed"] == 42 for c in config_list)
  assert all(c["new_param"] == "value" for c in config_list)


def test_piter_multiple_iterations():
  """Test that piter can be iterated multiple times with caching."""
  configs = piter({"lr": [0.001, 0.01]})
  
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
  configs = piter({
      "optimizer": ["adam", "sgd"],
      "config": [{"momentum": 0.9}, {"momentum": 0.95}]
  })
  
  config_list = list(configs)
  assert len(config_list) == 4
  assert config_list[0] == {"optimizer": "adam", "config": {"momentum": 0.9}}


def test_sweep_operators_return_piter():
  """Test that Sweep operators return ParameterIterator."""
  
  @proto
  class Config:
    lr: float = 0.001
  
  sweep = Sweep(Config)
  with sweep.product:
    Config.lr = [0.001, 0.01]
  
  # Test * operator
  result = sweep * piter({"batch_size": [32, 64]})
  assert isinstance(result, ParameterIterator)
  assert len(list(result)) == 4
  
  # Test % operator
  result = sweep % {"seed": 42}
  assert isinstance(result, ParameterIterator)
  assert all(c["seed"] == 42 for c in result)
  
  # Test ** operator
  result = sweep ** 2
  assert isinstance(result, ParameterIterator)
  assert len(list(result)) == 4  # 2 configs × 2 repeats


def test_piter_preserves_types():
  """Test that piter preserves value types."""
  configs = piter({
      "lr": [0.001, 0.01],  # float
      "batch_size": [32, 64],  # int
      "name": ["exp1", "exp2"],  # str
      "enabled": [True, False],  # bool
  })
  
  first_config = next(iter(configs))
  assert isinstance(first_config["lr"], float)
  assert isinstance(first_config["batch_size"], int)
  assert isinstance(first_config["name"], str)
  assert isinstance(first_config["enabled"], bool)


def test_piter_large_cartesian_product():
  """Test piter with large Cartesian product (memory efficiency)."""
  # This would create 1000 configs if materialized
  configs = piter({
      "param1": range(10),
      "param2": range(10),
      "param3": range(10),
  })
  
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
  # Define base experiments
  models = piter({"model": ["resnet18", "vit"]})
  datasets = piter({"dataset": ["cifar10", "imagenet"]})
  
  # Hyperparameters
  lr_schedule = piter({
      "lr": [0.001, 0.01],
      "lr_decay": [0.1, 0.5]
  })
  
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
  
  # Create piter configs
  piter_configs = piter({
      "config.lr": [0.001, 0.01],
      "config.batch_size": [32, 64]
  })
  
  # Convert to list and save manually
  configs_list = piter_configs.to_list()
  
  import json
  sweep_file = tmp_path / "piter_sweep.jsonl"
  with open(sweep_file, "w") as f:
    for config in configs_list:
      f.write(json.dumps(config) + "\n")
  
  # Load via Sweep
  sweep = Sweep(Config).load(str(sweep_file))
  loaded_configs = list(sweep)
  
  assert len(loaded_configs) == 4
  assert loaded_configs[0] == {"config.lr": 0.001, "config.batch_size": 32}


def test_piter_mod_with_nested_structure():
  """Test % operator with complex nested overrides."""
  configs = piter({"lr": [0.001, 0.01]})
  
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
  configs = piter({"lr": [0.001, 0.01]})
  
  with pytest.raises(ValueError, match="positive integer"):
    configs ** 0


def test_piter_pow_with_one():
  """Test ** operator with one (identity)."""
  configs = piter({"lr": [0.001, 0.01]})
  repeated = configs ** 1
  
  config_list = list(repeated)
  assert len(config_list) == 2
  assert config_list[0] == {"lr": 0.001}
  assert config_list[1] == {"lr": 0.01}


def test_piter_special_characters_in_keys():
  """Test piter with special characters in parameter names."""
  configs = piter({
      "model.conv.depth": [18, 50],
      "training.optimizer.lr": [0.001, 0.01],
      "data/path": ["/path1", "/path2"]
  })
  
  config_list = list(configs)
  assert len(config_list) == 8
  assert "model.conv.depth" in config_list[0]
  assert "data/path" in config_list[0]
