from params_proto.neo_proto import ParamsProto, get_children


def test_class():
    """The class should be usable as a namespace directly."""

    class Root(ParamsProto, prefix='root'):
        launch_type = 'borg'

    assert vars(Root) == {'launch_type': 'borg'}
    assert Root.launch_type == "borg"
    r = Root()
    assert r.launch_type == "borg"

    # Note: If you override the value on the master class
    #  it gets updated and propagates to the instance.
    Root.launch_type = "others"
    assert Root.launch_type == "others"
    r = Root()
    assert r.launch_type == "others"

    r = Root({"root.launch_type": 'local'})
    assert r.launch_type == "local"


def test_dependency():
    class Root(ParamsProto, prefix='root'):
        launch_type = 'borg'

    class SomeConfig(ParamsProto, prefix='resource'):
        fixed = "default"
        some_item = "default_value"

        @get_children
        def __init__(self, _deps=None, **children):
            root = Root(_deps)  # this updates the root object.
            if root.launch_type == "borg":
                self.some_item = "new_value"
            else:
                self.some_item = self.__class__.some_item

    s = SomeConfig({"root.launch_type": 'local'})
    assert s.some_item == "default_value"

    s = SomeConfig({"root.launch_type": 'borg'})
    assert s.some_item == "new_value"


def test_prefix():
    class Root(ParamsProto, prefix='root'):
        launch_type = 'borg'

    class Teacher(ParamsProto, prefix="resources.teacher"):
        cell = None
        autopilot = False

        @get_children
        def __init__(self, _deps, **children):
            if Root(_deps).launch_type != 'local':
                self.replicas_hint = children.get('replicas_hint', 26)
            super().__init__(**children)

    class Resources(ParamsProto):

        @get_children
        def __init__(self, _deps=None, **children):
            r = Root(_deps)
            print(children)
            self.item = children.get('teacher', None)
            self.teacher = Teacher(_deps, replicas_hint=26 if r.launch_type == 'borg' else 1)
            self.bad_teacher = Teacher(_deps, _prefix="bad_teacher")

    sweep_param = {
        "root.launch_type": "local",
        "resources.teacher.replica_hint": 10,
    }

    gd = Resources(sweep_param)
    assert set(vars(gd).keys()) == {"item", "teacher", "bad_teacher"}
    assert vars(gd.teacher) == {'cell': None, 'autopilot': False, 'replicas_hint': 1}
    assert vars(gd.bad_teacher) == {'cell': None, 'autopilot': False}
    assert gd.bad_teacher.cell is None
    assert gd.bad_teacher.autopilot is False
