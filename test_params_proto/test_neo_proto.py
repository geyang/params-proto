def test_simple_prefix():
    from params_proto.neo_proto import ParamsProto, get_children

    class Root(ParamsProto):
        _prefix = "not_root"

    assert Root._prefix == "not_root"
    assert Root(_prefix='yo')._prefix == "yo"

    class Root2(ParamsProto, prefix="this"):
        pass

    assert Root2._prefix == "this"
    assert Root(_prefix='yo')._prefix == "yo"


def test_namespace():
    """The class should be usable as a namespace directly. This
    would be the singleton pattern:

    We use a global configuration namespace, and dynamically
    override this namespace for each experiment.

    This is the most clean pattern, but the issue is that if you
    have dependencies, you won't be able to dynamically re-compute
    the dependent attributes.
    """
    from params_proto.neo_proto import ParamsProto, get_children

    # prefix is for the argparse (not implemented).
    class Root(ParamsProto, prefix='root'):
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

    from params_proto.neo_proto import ParamsProto, get_children

    class Root(ParamsProto, prefix='root'):
        launch_type = 'borg'

    class SomeConfig(ParamsProto, prefix='resource'):
        fixed = "default"
        some_item = "default_value"

        @get_children
        def __init__(self, _deps=None, **children):
            root = Root(_deps)  # this pulls the updated root.
            if root.launch_type == "borg":
                self.some_item = "new_value"
            else:
                self.some_item = self.__class__.some_item

    s = SomeConfig({"root.launch_type": 'local'})
    assert s.some_item == "default_value"

    s = SomeConfig({"root.launch_type": 'borg'})
    assert s.some_item == "new_value"


def test_prefix():
    """Testing """
    from params_proto.neo_proto import ParamsProto, get_children

    class Root(ParamsProto, prefix='root'):
        launch_type = 'borg'

    class Teacher(ParamsProto, prefix="resources.teacher"):
        cell = None
        autopilot = False

        @get_children
        def __init__(self, _deps, **children):
            self.replicas_hint = 1 if Root(_deps).launch_type == 'local' else 26
            super().__init__(**children)

    class Resources(ParamsProto, prefix="resources"):
        class default(ParamsProto):
            replicas_hint = 1

        @get_children
        def __init__(self, _deps=None, **children):
            r = Root(_deps)  # you can use the updated root.
            self.item = children.get('teacher', None)
            self.teacher = Teacher(_deps, )
            self.bad_teacher = Teacher(_deps, _prefix="resources.bad_teacher")

    sweep_param = {
        "root.launch_type": "local",
        "resources.teacher.replicas_hint": 10,
    }

    r = Resources(sweep_param)
    assert r.teacher._prefix == "resources.teacher"
    assert r.bad_teacher._prefix == "resources.bad_teacher"
    # this is problematic--default does not exist.
    assert set(vars(r).keys()) == {"item", "default", "teacher", "bad_teacher"}
    # assert vars(r) ==

    # The attributes can either be subclasses of ParamsProto, or instances.
    assert issubclass(r.default, ParamsProto), "the default should be a class object, not an instance"
    assert isinstance(r.teacher, ParamsProto), "this is an instance."

    # calling vars always gives you nested dicts (still debating if this is a good thing)
    assert vars(r)['teacher'] == {'cell': None, 'autopilot': False, 'replicas_hint': 10}
    assert vars(r.bad_teacher) == {'cell': None, 'autopilot': False}

    assert r.teacher.replicas_hint == 10

    assert r.bad_teacher.cell is None
    assert r.bad_teacher.autopilot is False


def test_root_config():
    """
    For overrides, we should be able to directly modify the root configuration object.
    """
    from params_proto.neo_proto import ParamsProto, get_children

    class Root(ParamsProto, prefix="."):
        root_attribute = 10

    override = {"root_attribute": 11}
    r = Root(override)
    print(f"{vars(r)}")
    print(r.root_attribute)
    import sys; print(sys.executable)
    assert r.root_attribute == 11

# def test_singleton_overwrite():
#     """
#     For overrides, we should be able to directly modify the root configuration object.
#     """
#     from params_proto.neo_proto import ParamsProto, get_children
#
#     class Root(ParamsProto):
#         root_attribute = 10
#
#     Root.update(root_attribute=11)
#     # r = Root(override)
#     assert Root.root_attribute == 11
