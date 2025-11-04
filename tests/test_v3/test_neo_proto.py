import os

from params_proto.v2 import ParamsProto, PrefixProto, Proto, get_children


def test_simple_prefix():
  class Root(ParamsProto, cli=False):
    _prefix = "not_root"

  assert Root._prefix == "not_root"
  assert Root(_prefix="yo")._prefix == "yo"

  class Root2(ParamsProto, prefix="this", cli=False):
    pass

  assert Root2._prefix == "this"
  assert Root(_prefix="yo")._prefix == "yo"


def test_update_no_prefix():
  class G(ParamsProto, cli=False):
    seed = 10

  d = {"seed": 20, "non_exist_gets_written": 10, "cascade.not_written": 30}
  G._update(d)
  assert G.seed == 20
  assert G.non_exist_gets_written == 10
  assert not hasattr(G, "cascade")


def test_update_with_prefix():
  class G(ParamsProto, cli=False, prefix=True):
    seed = 10

  d = {"G.seed": 20}
  G._update(d)
  assert G.seed == 20


def test_update_by_key():
  class G(ParamsProto, cli=False):
    seed = 10

  G._update(seed=20)
  assert G.seed == 20


def test_update_by_object_directly():
  class G(ParamsProto, cli=False, prefix=True):
    seed = 10

  g_updated = G(seed=20)

  assert vars(g_updated) == {"seed": 20}

  g_updated = G(seed=30)
  # vars(g) has no prefix, which is why we need to expand w/ **vars()
  G._update(**vars(g_updated))
  assert G.seed == 30


def test_namespace():
  """The class should be usable as a namespace directly. This
  would be the singleton pattern:

  We use a global configuration namespace, and dynamically
  override this namespace for each experiment.

  This is the most clean pattern, but the issue is that if you
  have dependencies, you won't be able to dynamically re-compute
  the dependent attributes.
  """

  # prefix is for the argparse (not implemented).
  class Root(ParamsProto, cli=False, prefix="root"):
    launch_type = "borg"

  assert Root._prefix == "root"
  assert vars(Root) == {"launch_type": "borg"}
  assert Root.launch_type == "borg"

  r = Root(_prefix="other")
  assert r._prefix == "other"

  # now test call the constructor.
  r = Root()
  assert r.launch_type == "borg", "the instance has the original value"

  # updating the class default before the construction affects instance.
  Root.launch_type = "others"
  r = Root()
  assert r.launch_type == "others", "the instance should get the updated values."

  # updating the class default after the construction does not affect the instance
  r = Root()
  Root.launch_type = "after"
  assert r.launch_type != "after", "the instance should not get the new value."

  # Update the configuration using the constructor.
  # This has two benefits:
  #  1. it is clear that you are always getting a new instance. `update()` calls
  #     mutates the instance in-place, and is undesirable.
  #  ~
  #  2. it allows hierarchical injection of local configurations.
  r = Root({"root.launch_type": "local"})
  assert r.launch_type == "local"

  # of course, you can be more direct:
  r = Root(**{"launch_type": "Sorry David, I can not do that."})
  assert r.launch_type == "Sorry David, I can not do that."

  # or simpler:
  r = Root(launch_type="check out: pip install jaynes")
  assert r.launch_type == "check out: pip install jaynes"


def test_dependency():
  """ParamsProto should allow levels of dependencies."""

  class Root(ParamsProto, cli=False, prefix="root"):
    launch_type = "borg"

  class SomeConfig(ParamsProto, cli=False, prefix="resource"):
    fixed = "default"
    some_item = "default_value"

    @get_children
    def __init__(self, _deps=None, **children):
      super().__init__(_deps, **children)

      root = Root(_deps)  # this pulls the updated root.
      if root.launch_type == "borg":
        self.some_item = "new_value"
      else:
        self.some_item = SomeConfig.some_item

  s = SomeConfig({"root.launch_type": "local"})
  assert s.some_item == "default_value"

  s = SomeConfig({"root.launch_type": "borg"})
  assert s.some_item == "new_value"


def test_prefix():
  """Testing"""

  class PPO(ParamsProto, cli=False, prefix="PPO"):
    num_envs = 10

    # Right now this is not automatic.
    class sim(ParamsProto, cli=False, prefix="PPO.sim"):
      ptype = "physx"
      decimation = 4

  sweep_param = {
    "PPO.num_envs": 5,
    "PPO.sim.decimation": 2,
  }

  assert PPO.sim._prefix == "PPO.sim", "the prefix should be correct."

  cfg = PPO(sweep_param)
  assert cfg.num_envs == 5
  assert cfg.sim.decimation == 2

  # now test the nested dictionary fromt he config file
  assert cfg._tree == {"num_envs": 5, "sim": {"decimation": 2, "ptype": "physx"}}

  assert PPO._tree == {"num_envs": 10, "sim": {"decimation": 4, "ptype": "physx"}}


def test_root_config():
  """
  For overrides, we should be able to directly modify the root configuration object.
  """

  class Root(ParamsProto, cli=False, prefix="."):
    root_attribute = 10

  override = {"root_attribute": 11}
  r = Root(override)
  print(f"{vars(r)}")
  print(r.root_attribute)
  import sys

  print(sys.executable)
  assert r.root_attribute == 11


# noinspection PyPep8Naming
def test_Proto_default():
  a = Proto(default=10)
  assert a.default == 10, "default should be correct"
  assert a.value == 10, "value should default to the original value"

  class Root(ParamsProto, cli=False, prefix="."):
    root_attribute = Proto(default=10)
    other_1 = Proto(20, "this is help text")

  print(vars(Root))
  assert vars(Root) == {"other_1": 20, "root_attribute": 10}
  assert Root.root_attribute == 10
  assert Root().root_attribute == 10

  Root.root_attribute = 20
  assert Root.root_attribute == 20
  r = Root()
  r.root_attribute = 30
  assert r.root_attribute == 30


# noinspection PyPep8Naming
def test_Proto_env():
  class Root(ParamsProto, cli=False, prefix="."):
    home = Proto(default="default", env="HOME")
    home_and_some = Proto(default="default", env="$HOME/and_some")

  assert Root.home == os.environ["HOME"]
  assert Root.home_and_some == os.environ["HOME"] + "/and_some"


# noinspection PyPep8Naming
def test_Proto_env_priority():
  class Root(ParamsProto, cli=False, prefix="."):
    home = Proto(default="default", env="DOES_NOT_EXIST")
    dollar_sign = Proto(default="default", env="$DOES_NOT_EXIST")
    dollar_sign_complex = Proto(default="default", env="$DOES_NOT_EXIST/and_some")

  assert Root.home == "default"
  assert Root.dollar_sign == "default"
  assert Root.dollar_sign_complex == "default"


def test_none_overwrite():
  """The point of this test is to make sure None values also gets written."""

  class A(ParamsProto, cli=False, prefix=True):
    key = 10

  A._update({"A.key": None})
  assert A.key is None, "key should not be `None`."


# def test_singleton_overwrite():
#     """
#     For overrides, we should be able to directly modify the root configuration object.
#     """
#     from params_proto.neo_proto import ParamsProto, get_children
#
#     class Root(ParamsProto, cli=False):
#         root_attribute = 10
#
#     Root.update(root_attribute=11)
#     # r = Root(override)
#     assert Root.root_attribute == 11


def test_deep_nested():
  """The point of this test is to test nested protos."""
  from params_proto.v2.proto import PrefixProto

  class A(PrefixProto, cli=False):
    key = 10

    class B(PrefixProto, cli=False):
      key = 20

      class C(PrefixProto, cli=False):
        key = 30

  A._update({"A.key": None, "A.B.key": "hey", "A.B.C.key": "yo"})

  assert A.key is None, "key should not be `None`."
  assert A.B.key == "hey", "key should be `hey`."
  assert A.B.C.key == "yo", "key should be `yo`."


def test_inheritance():
  """
  The point of this test is to make sure that the inheritance works.
  """

  class Root:
    root_name: str = "root"

    def a_method_should_not_appear(self):
      return "should NOT appear"

    @staticmethod
    def a_static_method_should_not_appear():
      return "should NOT appear"

    @property
    def custom_property(self):
      return "custom_property works"

  class Parent(Root):
    parent_name: str = "parent"

    @property
    def parent_property(self):
      return "parent_property works"

  class Args(ParamsProto, Parent):
    seed: int = 100
    text: str = "hello"

    def self_method(self):
      return "has self"

    @property
    def args_property(self):
      return "args_property works"

    def __post_init__(self):
      print("Args.__post_init__")

  assert Args.root_name == "root"
  assert Args.parent_name == "parent"
  assert Args.custom_property.__get__(Args) == "custom_property works"
  assert Args.args_property.__get__(Args) == "args_property works"
  assert Args.a_method_should_not_appear(Args) == "should NOT appear"
  assert Args.a_static_method_should_not_appear() == "should NOT appear"
  # this is used during initialization
  assert Args.__vars__ == {
    "root_name": "root",
    "parent_name": "parent",
    "parent_property": Args.parent_property,
    "seed": 100,
    "text": "hello",
    "args_property": Args.args_property,
    "custom_property": Args.custom_property,
  }
  assert vars(Args) == {
    "root_name": "root",
    "parent_name": "parent",
    "parent_property": "parent_property works",
    "seed": 100,
    "text": "hello",
    "args_property": "args_property works",
    "custom_property": "custom_property works",
  }

  Root.root_name = "new_root"
  assert Args.root_name == "new_root", "should update."

  args = Args()
  (
    args.a_method_should_not_appear() == "should NOT appear",
    "should be able to call the method.",
  )
  args.self_method() == "has self", "the method should have access to self."


def test_instance_inheritance():
  """
  The point of this test is to make sure that the inheritance works.
  """

  class Root:
    root_name: str = "root"

    def __post_init__(self):
      print("\nRoot.__post_init__")

    def a_method_should_not_appear(self):
      return "should NOT appear"

    @staticmethod
    def a_static_method_should_not_appear():
      return "should NOT appear"

    @property
    def custom_property(self):
      return "custom_property works"

  class Parent(Root):
    parent_name: str = "parent"

    def __post_init__(self):
      super().__post_init__()
      print("Parent.__post_init__")

    @property
    def parent_property(self):
      return "parent_property works"

  class Args(PrefixProto, Parent):
    seed: int = 100
    text: str = "hello"

    @property
    def args_property(self):
      return "args_property works"

    def __post_init__(self):
      Parent.__post_init__(self)
      print("Args.__post_init__")

  args = Args()
  Root.root_name = "new_root"

  assert vars(args) == {
    "root_name": "root",
    "parent_name": "parent",
    "parent_property": "parent_property works",
    "seed": 100,
    "text": "hello",
    "args_property": "args_property works",
    "custom_property": "custom_property works",
  }
  assert args.root_name == "root"
  assert args.parent_name == "parent"
  assert args.custom_property == "custom_property works"
  assert args.args_property == "args_property works"


def test_dict_attr():
  """dict attr used to return Bear"""
  from waterbear import Bear

  class Args(ParamsProto, cli=False):
    dict_attr = {}

    def instance_method(
      self,
    ):
      assert isinstance(self, Args), "self should be an instance of Args"

  assert not isinstance(Args.dict_attr, Bear), "should not be Bear"

  args = Args()
  assert not isinstance(args.dict_attr, Bear), "should not be Bear"


def test_class_property():
  """dict attr used to return Bear"""

  class Args(ParamsProto, cli=False):
    dict_attr = {}

    @property
    def some_attr(
      self,
    ):
      return "attr value"

  assert Args().some_attr == "attr value"
  # assert not isinstance(Args.dict_attr, Bear), "should not be Bear"
  #
  # args = Args()
  # assert not isinstance(args.dict_attr, Bear), "should not be Bear"


def test_instance_method():
  """dict attr used to return Bear"""
  from types import MethodType

  class Args(ParamsProto, cli=False):
    def instance_method(
      self,
    ):
      assert isinstance(self, Args), "self should be an instance of Args"

    arrow_fn = lambda self: self

  args = Args()
  args.instance_method()
  assert isinstance(args.instance_method, MethodType), "the functions should be bounded"

  new_self = args.arrow_fn()
  assert isinstance(new_self, Args), "self should be an instance of Args"
  assert isinstance(args.arrow_fn, MethodType), "the arrow_fn should also be bounded"
