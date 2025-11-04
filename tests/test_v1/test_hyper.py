import pytest

from params_proto.v1.hyper import Sweep, dot_join
from params_proto.v2 import ARGS, PrefixProto


def test_dot_join():
  assert dot_join(None, None) is None


@pytest.fixture
def clear_args():
  ARGS.clear()


def test_set(clear_args):
  sweep = Sweep()  # create a new sweep instance

  # usage
  class G(PrefixProto):
    start_seed = 10
    discern_flag = True
    env_name = "dm_lab"
    batch_size = 5
    replicas_hint = ["yo", "hey"]

  with sweep.set(G) as _:
    _.start_seed = 10
    _.discern_flag = False

  all = [*sweep]
  assert all == [{"G.discern_flag": False, "G.start_seed": 10}]


def test_set_advanced(clear_args):
  sweep = Sweep()  # create a new sweep instance

  # usage
  class G(PrefixProto):
    start_seed = 10
    discern_flag = True
    env_name = "dm_lab"
    batch_size = 5
    replicas_hint = ["yo", "hey"]

  with sweep.set(G) as _:
    _.replicas_hint = 12

    with sweep.zip(G) as _:
      _.env_name = ["small", "large"]
      _.batch_size = [10, 50]

  all = [*sweep]
  assert len(all) == 2


def test_set_advanced_2(clear_args):
  sweep = Sweep()  # create a new sweep instance

  # usage
  class G(PrefixProto):
    null_attribute = True
    start_seed = 10
    discern_flag = True
    env_name = "dm_lab"
    batch_size = 5
    replicas_hint = ["yo", "hey"]

  with sweep.set(G) as _:
    _.null_attribute = None
    _.replicas_hint = 12

    with sweep.product(G) as _:
      _.start_seed = range(15)

      with sweep.zip(G) as _:
        _.env_name = ["small", "large", "xl"]
        _.batch_size = [10, 50, 100]

  all = [*sweep]
  assert len(all) == 45


def test_product(clear_args):
  sweep = Sweep()  # create a new sweep instance

  # usage
  class G(PrefixProto):
    start_seed = 10
    discern_flag = True
    env_name = "dm_lab"
    batch_size = 5
    replicas_hint = ["yo", "hey"]

  with sweep.product(G) as _:
    _.start_seed = range(2)
    _.discern_flag = [True, False]

  all = [*sweep]
  assert all == [
    {"G.discern_flag": True, "G.start_seed": 0},
    {"G.discern_flag": False, "G.start_seed": 0},
    {"G.discern_flag": True, "G.start_seed": 1},
    {"G.discern_flag": False, "G.start_seed": 1},
  ]


def test_zip(clear_args):
  sweep = Sweep()  # create a new sweep instance

  # usage
  class G(PrefixProto):
    start_seed = 10
    discern_flag = True
    env_name = "dm_lab"
    batch_size = 5
    replcas_hint = ["yo", "hey"]

  with sweep.zip(G) as _G:
    _G.env_name = ["small_env", "large_env"]
    _G.batch_size = [10, 50]

  all = [*sweep]
  assert all == [
    {"G.batch_size": 10, "G.env_name": "small_env"},
    {"G.batch_size": 50, "G.env_name": "large_env"},
  ]


def test_product_zip(clear_args):
  sweep = Sweep()  # create a new sweep instance

  class G(PrefixProto):
    start_seed = 10
    discern_flag = True
    env_name = "dm_lab"
    batch_size = 5
    replcas_hint = ["yo", "hey"]

  with sweep.product(G) as _:
    _.start_seed = range(2)
    _.discern_flag = [True, False]

    with sweep.zip(G) as _:
      _.env_name = ["small_env", "large_env"]
      _.batch_size = [10, 50]

  all = [*sweep]
  assert len(all) == 2 * 2 * 2
  assert all == [
    {
      "G.batch_size": 10,
      "G.discern_flag": True,
      "G.env_name": "small_env",
      "G.start_seed": 0,
    },
    {
      "G.batch_size": 50,
      "G.discern_flag": True,
      "G.env_name": "large_env",
      "G.start_seed": 0,
    },
    {
      "G.batch_size": 10,
      "G.discern_flag": False,
      "G.env_name": "small_env",
      "G.start_seed": 0,
    },
    {
      "G.batch_size": 50,
      "G.discern_flag": False,
      "G.env_name": "large_env",
      "G.start_seed": 0,
    },
    {
      "G.batch_size": 10,
      "G.discern_flag": True,
      "G.env_name": "small_env",
      "G.start_seed": 1,
    },
    {
      "G.batch_size": 50,
      "G.discern_flag": True,
      "G.env_name": "large_env",
      "G.start_seed": 1,
    },
    {
      "G.batch_size": 10,
      "G.discern_flag": False,
      "G.env_name": "small_env",
      "G.start_seed": 1,
    },
    {
      "G.batch_size": 50,
      "G.discern_flag": False,
      "G.env_name": "large_env",
      "G.start_seed": 1,
    },
  ]
