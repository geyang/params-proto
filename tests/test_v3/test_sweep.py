"""Tests for Sweep with v3 @proto decorator API."""
import pytest

from params_proto import proto, Sweep


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
        with sweep.set:
            Training.nonexist_gets_written = "hey"
        with sweep.product:
            Training.b = range(100)

    for i, deps in enumerate(sweep):
        assert deps == {"training.b": i, "training.nonexist_gets_written": "hey"}
        assert Training.b == i
        assert Training.nonexist_gets_written == "hey"


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
        Config.some_value = "random-string"
        assert Config.some_value == "random-string", (
            "should be able to get the updated value right away"
        )

    all_configs = list(sweep)

    assert all_configs[0]["config.postfix"] == "config.seed-(10)"
    assert all_configs[1]["config.postfix"] == "config.seed-(20)"
    assert all_configs[2]["config.postfix"] == "config.seed-(30)"
    assert all_configs[0]["config.some_value"] == "random-string"


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
    sweep.save(str(sweep_file), verbose=False)

    # Load from file
    loaded_sweep = Sweep(Config).load(str(sweep_file))
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
    @proto
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
