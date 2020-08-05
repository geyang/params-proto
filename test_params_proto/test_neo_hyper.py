from params_proto.neo_hyper import Sweep, dot_join
from params_proto.neo_proto import ParamsProto


def test_dot_join():
    assert dot_join(None, None) is None


def test_setter_and_getter_hook():
    class G(ParamsProto):
        start_seed = 10
        discern_flag = True
        env_name = "dm_lab"
        batch_size = 5
        replicas_hint = ['yo', 'hey']

    def setter(Proto, item, value):
        print(Proto, item, value)

    print(vars(G))
    G._add_hook(setter)

    assert G.start_seed == 10
    G.start_seed = 20
    G._pop_hooks()


def test_multiple_configs():
    class G(ParamsProto):
        a = 5

    class DEBUG(ParamsProto):
        b = 10

    with Sweep(G, DEBUG) as sweep:
        with sweep.zip:
            G.a = range(100)
            DEBUG.b = range(100)

    for i, all_deps in enumerate(sweep):
        assert G.a == i
        assert DEBUG.b == i

    # only one still works
    with Sweep(G, DEBUG) as sweep:
        with sweep.set:
            DEBUG.prefix = "hey"
        with sweep.product:
            DEBUG.b = range(100)

    # note: does not support "_" (underscore) prefix.
    for i, deps in enumerate(sweep):
        assert deps == {"DEBUG.b": i, "DEBUG.prefix": "hey"}
        assert DEBUG.b == i
        assert DEBUG.prefix == "hey"


def test_incrementation():
    """The Sweep resets the configuration at each step,
    to make sure that local overrides do not propagate
    to the next step. This also means that you can not
    imperatively mutate the value step-by-step, such
    as incrementing a counter.

    There are a few patterns for accomplishing this.
    """
    from params_proto.neo_proto import Accumulant

    class G(ParamsProto):
        static_counter = 10
        dynamic_accumulant = Accumulant(10 - 1)

        @classmethod
        def __init__(cls, ):
            cls.static_counter += 1
            cls.dynamic_counter = getattr(cls, "dynamic_counter", -1) + 1
            cls.dynamic_accumulant += 1

    with Sweep(G) as sweep:
        with sweep.product:
            G.seed = [i for i in range(10)]

    for deps in sweep:
        G()
        assert G.static_counter == 11
        assert G.dynamic_counter == G.seed
        assert G.dynamic_accumulant == 10 + G.seed


def test_subscription():
    class G(ParamsProto):
        start_seed = 10

    with Sweep(G) as sweep:
        with sweep.product:
            G.start_seed = list(range(100))

    # using sweep as a sliced generator.
    assert list(sweep[:5]) == [{"G.start_seed": i} for i in range(5)]
    assert list(sweep[10:20:3]) == [{"G.start_seed": i} for i in range(10, 20, 3)]
    assert list(sweep[30]) == [{"G.start_seed": 30}]
    assert list(sweep[1:]) == [{"G.start_seed": i} for i in range(1, 100)]


def test_negative_subscription():
    class G(ParamsProto):
        start_seed = 10

    with Sweep(G) as sweep:
        with sweep.product:
            G.start_seed = list(range(100))

    # using sweep as a sliced generator.
    for a, b in zip(list(sweep[-10:-5]), [{"G.start_seed": i} for i in range(90, 95)]):
        assert a == b


def test_product():
    # usage
    class G(ParamsProto):
        start_seed = 10
        discern_flag = True
        env_name = "dm_lab"
        batch_size = 5
        replicas_hint = ['yo', 'hey']

    with Sweep(G) as sweep:
        with sweep.product:
            G.start_seed = range(2)
            G.discern_flag = [True, False]

    all = [*sweep]
    assert all == [{'G.discern_flag': True, 'G.start_seed': 0},
                   {'G.discern_flag': False, 'G.start_seed': 0},
                   {'G.discern_flag': True, 'G.start_seed': 1},
                   {'G.discern_flag': False, 'G.start_seed': 1}]


def test_zip():
    # usage
    class G(ParamsProto):
        start_seed = 10
        discern_flag = True
        env_name = "dm_lab"
        batch_size = 5
        replcas_hint = ['yo', 'hey']

    with Sweep(G) as sweep:
        with sweep.zip:
            G.env_name = ['small_env', 'large_env']
            G.batch_size = [10, 50]

    all = [*sweep]
    assert all == [{'G.batch_size': 10, 'G.env_name': 'small_env'},
                   {'G.batch_size': 50, 'G.env_name': 'large_env'}]


def test_product_zip():
    class G(ParamsProto):
        start_seed = 10
        discern_flag = True
        env_name = "dm_lab"
        batch_size = 5
        replcas_hint = ['yo', 'hey']

    with Sweep(G) as sweep:
        with sweep.product:
            G.start_seed = range(2)
            G.discern_flag = [True, False]

            with sweep.zip:
                G.env_name = ['small_env', 'large_env']
                G.batch_size = [10, 50]

    all = [*sweep]
    assert len(all) == 2 * 2 * 2
    assert all == [{'G.batch_size': 10, 'G.discern_flag': True, 'G.env_name': 'small_env', 'G.start_seed': 0},
                   {'G.batch_size': 50, 'G.discern_flag': True, 'G.env_name': 'large_env', 'G.start_seed': 0},
                   {'G.batch_size': 10, 'G.discern_flag': False, 'G.env_name': 'small_env', 'G.start_seed': 0},
                   {'G.batch_size': 50, 'G.discern_flag': False, 'G.env_name': 'large_env', 'G.start_seed': 0},
                   {'G.batch_size': 10, 'G.discern_flag': True, 'G.env_name': 'small_env', 'G.start_seed': 1},
                   {'G.batch_size': 50, 'G.discern_flag': True, 'G.env_name': 'large_env', 'G.start_seed': 1},
                   {'G.batch_size': 10, 'G.discern_flag': False, 'G.env_name': 'small_env', 'G.start_seed': 1},
                   {'G.batch_size': 50, 'G.discern_flag': False, 'G.env_name': 'large_env', 'G.start_seed': 1}]


def test_set():
    class G(ParamsProto):
        start_seed = 10
        discern_flag = True
        env_name = "dm_lab"
        batch_size = 5
        replicas_hint = ['yo', 'hey']

    with Sweep(G) as sweep:
        G.start_seed = 20
        G.discern_flag = False

    for override in sweep:
        assert G.discern_flag is False
        assert override == {'G.discern_flag': False, 'G.start_seed': 20}


def test_set_advanced():
    # usage
    class G(ParamsProto):
        start_seed = 10
        discern_flag = True
        env_name = "dm_lab"
        batch_size = 5
        replicas_hint = ['yo', 'hey']

    with Sweep(G) as sweep:
        G.replicas_hint = 12

        with sweep.zip:
            G.env_name = ['small', 'large']
            G.batch_size = [10, 50]

    all = [*sweep]
    assert len(all) == 2


def test_set_advanced_2():
    # usage
    class G(ParamsProto):
        null_attribute = True
        start_seed = 10
        discern_flag = True
        env_name = "dm_lab"
        batch_size = 5
        replicas_hint = ['yo', 'hey']

    with Sweep(G) as sweep:
        G.null_attribute = None
        G.replicas_hint = 12

        with sweep.product:
            G.start_seed = range(15)

            with sweep.zip:
                G.env_name = ['small', 'large', 'xl']
                G.batch_size = [10, 50, 100]

    all = [*sweep]
    assert len(all) == 45


def test_chaining():
    # usage
    class G(ParamsProto):
        level_name = "dmlab"
        some_prefix = f"{level_name}/1"
        start_seed = 10

    with Sweep(G) as sweep:
        G.level_name = "gotham"
        G.some_prefix = f"gotham/1"
        with sweep.product:
            G.start_seed = range(15)

    with sweep.set:
        G.level_name = "dmlab"
        G.some_prefix = f"dmlab/1"
        with sweep.product:
            G.start_seed = range(15)

    all = [*sweep]
    print(len(all))
    assert len(all) == 30


def test_chaining_with_shared_root_set():
    # usage
    class G(ParamsProto):
        root_set = False
        level_name = "dmlab"
        some_prefix = f"{level_name}/1"
        start_seed = 10

    with Sweep(G) as sweep:
        G.root_set = True

        with sweep.chain:
            with sweep.set:
                G.level_name = "gotham"
                G.some_prefix = f"gotham/1"

                with sweep.product:
                    G.start_seed = range(15)

            with sweep.set:
                G.level_name = "dmlab"
                G.some_prefix = f"dmlab/1"

                with sweep.product:
                    G.start_seed = range(15)

    all = [*sweep]
    assert len(all) == 30


def test_jagged():
    """the point of this test is to make sure different config with different keys
    always rewrite from the original."""

    # usage
    class G(ParamsProto):
        config_1 = False
        config_2 = False

    with Sweep(G) as sweep:

        with sweep.chain:
            with sweep.set:
                G.config_1 = 10
            with sweep.set:
                G.config_2 = 20

    for i, deps in enumerate(sweep):
        if i == 0:
            assert G.config_1 == 10
            assert G.config_2 is False
        if i == 1:
            assert G.config_1 is False
            assert G.config_2 == 20


def test_each():
    """Can register a function to be ran for each configuration. Useful for
    setting values that dynamically depend on other.s"""

    class G(ParamsProto):
        seed = 10
        postfix = "seed-"

    with Sweep(G).product as sweep:
        G.seed = [10, 20, 30]

    def each(G):
        G.postfix = f"G.seed-({G.seed})"

    sweep.each(each)

    all = list(sweep)

    assert all[0]['G.postfix'] == "G.seed-(10)"
    assert all[1]['G.postfix'] == "G.seed-(20)"
    assert all[2]['G.postfix'] == "G.seed-(30)"
