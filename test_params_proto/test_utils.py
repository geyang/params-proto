from params_proto.utils import dot_to_deps


def test_empty():
    empty = {}
    _ = dot_to_deps(empty)
    assert _ == {}, "result should be empty"


def test_dot():
    set_self = {".": 10}  # should preserve
    _ = dot_to_deps(set_self)
    assert _ == {'.': 10}, "should contain dot"


def test_normal():
    _ = dot_to_deps({'root.launch_type': 'local'}, "root")
    assert _ == {"launch_type": 'local'}


def test_nested():
    set_bunch = {"resources.teacher.replica_hint": 10, }
    _ = dot_to_deps(set_bunch, 'resources')
    assert _ == {'teacher': {'replica_hint': 10}}, "should contain nested dict"


def test_dot_prefix():
    _ = dot_to_deps({'resources.teacher.replicas_hint': 10}, "resources.teacher")
    assert _ == {"replicas_hint": 10}


def test_multile_prefix():
    _ = dot_to_deps({'resources.teacher.replicas_hint': 10}, "resources", "teacher")
    assert _ == {"replicas_hint": 10}


def test_mixed_prefixes():
    _ = dot_to_deps({'root.resources.teacher.replicas_hint': 10}, "root.resources", "teacher")
    assert _ == {"replicas_hint": 10}
