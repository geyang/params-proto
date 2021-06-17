import pytest
import sys
from params_proto.neo_proto import ParamsProto, Proto, ARGS, Flag


@pytest.fixture
def single_config():
    old_argv = sys.argv.copy()
    for k, v in {
        "--env_name": "FetchPickAndPlace-v1",
        "--seed": "100",
    }.items():
        if k not in sys.argv:
            sys.argv.extend([k, v])
    ARGS.clear()
    yield
    sys.argv[:] = old_argv


@pytest.fixture
def flag_config():
    old_argv = sys.argv.copy()
    sys.argv.extend(["--some-feature"])
    ARGS.clear()
    yield
    sys.argv[:] = old_argv


@pytest.fixture
def prefixed_config():
    import sys
    old_argv = sys.argv.copy()
    for k, v in {
        "--Second.bool": "False",
        "--Second.env_name": "FetchPickAndPlace-v1",
        "--Second.seed": "100",
    }.items():
        if k not in sys.argv:
            sys.argv.extend([k, v])
    ARGS.clear()
    yield
    sys.argv[:] = old_argv


def test_argparse_override(single_config):
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default='FetchReach-v1')
    parser.add_argument('--seed', type=int, default=123)
    # parser.add_argument("args", nargs="+")
    # help = parser.format_help()
    # print(help)

    args, unkown = parser.parse_known_args()
    assert args.env_name == "FetchPickAndPlace-v1"
    assert args.seed == 100


def test_simple_cli_args(single_config):
    # todo: test default then test override.
    class Root(ParamsProto):
        """
        Root Configuration Object
        """
        env_name = "FetchReach-v1"
        seed = 123

    assert Root._prefix is None
    help = ARGS.parser.format_help()
    print(help)


def test_multiple_cli_args(prefixed_config):
    # todo: need to clear the ARGS command to isolate the
    #   changes for these tests

    class Root(ParamsProto, parse_args=False):
        """
        Root Configuration Object
        """
        env_name = "FetchReach-v1"
        seed = 123

    class Second(ParamsProto, prefix=True):
        """
        The Second Configuration Object
        """
        env_name = "FetchReach-v1"
        seed = 123
        bool = True

    print(">>>1", vars(Second))
    help = ARGS.parser.format_help()
    print(help)

    print(">>>2", Second.env_name)
    assert Second.env_name == "FetchPickAndPlace-v1"

    assert Second.bool is False


def test_bool_flags(flag_config):
    # todo: need to clear the ARGS command to isolate the
    #   changes for these tests

    class Root(ParamsProto):
        """Root Configuration Object with Flags

        The `Flag` primitive allows one to set an attribute
        to a specific value via `to_value` argument. `to_value`
        is default to True, whereas the default value of this
        `Flat` primitive is `None`.

        [Usage]

        ```python
        class Args(PrefixProto):
            some_feature = Flag(help="this is a feature flag for xxx",
                                to_value=True, default=None)
        ```

        The first argument of the `Flag` primitive is the help string.
        This is because in most cases, we can use the default `to_value=True`
        """
        env_name = "FetchReach-v1"
        seed = 123
        some_feature = Flag(help="This is a feature flag for xxx",
                            to_value="feature-is-on", default=None)

    print(">>>2", Root.some_feature)
    print(ARGS.parser.format_help())
    assert Root.some_feature == "feature-is-on"


def test_ENV_params(prefixed_config):
    # todo: need to clear the ARGS command to isolate the
    #   changes for these tests

    class Root(ParamsProto, parse_args=False):
        """
        Root Configuration Object
        """
        env_name = Proto("FetchReach-v1", env="ENV_NAME", help="this is a very long readme and it goes on and one and on and never stops. The line breaks have a large indent and it is not really clear how the indentation actually works. It almost looks like the paragraph is right aligned.")
        seed = Proto(123, help="this is short and longer")
        home = Proto('ge', env="USER", help="this is short and longer")

    help = ARGS.parser.format_help()
    print(help)
