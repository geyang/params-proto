import os

from params_proto import ParamsProto, get_children, Proto


def test_simple_prefix():
    class Root(ParamsProto, cli=False):
        _prefix = "not_root"

    assert Root._prefix == "not_root"
    assert Root(_prefix='yo')._prefix == "yo"

    class Root2(ParamsProto, prefix="this", cli=False):
        pass

    assert Root2._prefix == "this"
    assert Root(_prefix='yo')._prefix == "yo"


def test_update_no_prefix():
    class G(ParamsProto, cli=False):
        seed = 10

    d = {"seed": 20, "non_exist_gets_written": 10, "cascade.not_written": 30}
    G._update(d)
    assert G.seed == 20
    assert G.non_exist_gets_written == 10
    assert not hasattr(G, 'cascade')


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
    class Root(ParamsProto, cli=False, prefix='root'):
        launch_type = 'borg'

    assert Root._prefix == "root"
    assert vars(Root) == {'launch_type': 'borg'}
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
    r = Root({"root.launch_type": 'local'})
    assert r.launch_type == "local"

    # of course, you can be more direct:
    r = Root(**{"launch_type": 'Sorry David, I can not do that.'})
    assert r.launch_type == "Sorry David, I can not do that."

    # or simpler:
    r = Root(launch_type='check out: pip install jaynes')
    assert r.launch_type == "check out: pip install jaynes"


def test_dependency():
    """ParamsProto should allow levels of dependencies."""

    class Root(ParamsProto, cli=False, prefix='root'):
        launch_type = 'borg'

    class SomeConfig(ParamsProto, cli=False, prefix='resource'):
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

    s = SomeConfig({"root.launch_type": 'local'})
    assert s.some_item == "default_value"

    s = SomeConfig({"root.launch_type": 'borg'})
    assert s.some_item == "new_value"


def test_prefix():
    """Testing """

    class Root(ParamsProto, cli=False, prefix='root'):
        launch_type = 'borg'

    class Teacher(ParamsProto, cli=False, prefix="resources.teacher"):
        cell = None
        autopilot = False

        @get_children
        def __init__(self, _deps, **children):
            super().__init__(_deps, **children)
            print(children)

            self.replicas_hint = 1 if Root(_deps).launch_type == 'local' else 26

    t = Teacher()
    print(">>>", t, type(t))

    class Resources(ParamsProto, cli=False, prefix="resources"):
        class default(ParamsProto, cli=False):
            replicas_hint = 1

        @get_children
        def __init__(self, _deps=None, **children):
            super().__init__(_deps, **children)
            print(super().__init__)
            print('======================')

            r = Root(_deps)  # you can use the updated root.
            print(children)
            self.item = children.get('teacher', None)
            self.teacher = Teacher(_deps, )
            self.bad_teacher = Teacher(_deps, _prefix="resources.bad_teacher")

    sweep_param = {
        "root.launch_type": "local",
        "resources.teacher.replicas_hint": 10,
    }

    r = Resources(sweep_param)
    # assert r.teacher._prefix == "resources.teacher"
    # assert r.bad_teacher._prefix == "resources.bad_teacher"
    # # this is problematic--default does not exist.
    # assert set(r.__dict__.keys()) == {"item", "default", "teacher", "bad_teacher"}

    # The attributes can either be subclasses of ParamsProto, or instances.
    print(">>>", r.default, type(r.default))
    assert isinstance(r.default, Resources.default), "the default should be a class object, not an instance"
    # assert isinstance(r.teacher, ParamsProto), "this is an instance."
    #
    # # calling vars always gives you nested dicts (still debating if this is a good thing)
    # assert vars(r)['teacher'] == {'cell': None, 'autopilot': False, 'replicas_hint': 10}
    # assert vars(r.bad_teacher) == {'cell': None, 'autopilot': False}
    #
    # assert r.teacher.replicas_hint == 10
    #
    # assert r.bad_teacher.cell is None
    # assert r.bad_teacher.autopilot is False


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
    assert vars(Root) == {'other_1': 20, 'root_attribute': 10}
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
        home = Proto(default='default', env="HOME")
        home_and_some = Proto(default='default', env="$HOME/and_some")

    assert Root.home == os.environ['HOME']
    assert Root.home_and_some == os.environ['HOME'] + "/and_some"


def test_none_overwrite():
    """The point of this test is to make sure None values also gets written."""

    class A(ParamsProto, cli=False, prefix=True):
        key = 10

    A._update({'A.key': None})
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
    from params_proto.proto import PrefixProto

    class A(PrefixProto, cli=False):
        key = 10

        class B(PrefixProto, cli=False):
            key = 20

            class C(PrefixProto, cli=False):
                key = 30

    A._update({'A.key': None, 'A.B.key': 'hey', 'A.B.C.key': 'yo'})

    assert A.key is None, "key should not be `None`."
    assert A.B.key is "hey", "key should be `hey`."
    assert A.B.C.key is "yo", "key should be `yo`."


def test_inheritance():
    """
    The point of this test is to make sure that the inheritance works.
    """
    from params_proto import PrefixProto

    class Root:
        root_name: str = "root"

        @property
        def custom_property(self):
            return "custom_property works"

    class Parent(Root):
        parent_name: str = "parent"

    class Args(Parent, PrefixProto):
        seed: int = 100
        text: str = "hello"

        @property
        def args_property(self):
            return "args_property works"

        def __post_init__(self):
            print("Args.__post_init__")

    args = Args()
    assert args.root_name == "root"
    assert args.parent_name == "parent"
    assert args.custom_property == "custom_property works"
    assert args.args_property == 'args_property works'

    args_2 = Args(_deps={"Args.root_name": "new_root_name"})
    assert args_2.root_name == "new_root_name", "the root name should be updated"

    Root.root_name = "updated"
    args_3 = Args()
    assert args_3.root_name == "updated", "the root name should also update."

    assert Args.parent_name == "parent", "Args.parent_name should be 'root'"
