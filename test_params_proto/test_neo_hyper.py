from params_proto.neo_proto import ParamsProto
from params_proto.neo_hyper import Sweep, dot_join


def test_dot_join():
    assert dot_join(None, None) is None


def test_setter_and_getter_hook():
    # usage
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

