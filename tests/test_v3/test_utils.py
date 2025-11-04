from params_proto.v2.utils import dot_to_deps


def test_empty():
  empty = {}
  _ = dot_to_deps(empty)
  assert _ == {}, "result should be empty"


def test_dot():
  set_self = {".": 10}  # should preserve
  _ = dot_to_deps(set_self)
  assert _ == {".": 10}, "should contain dot"


def test_normal():
  _ = dot_to_deps({"root.launch_type": "local"}, "root")
  assert _ == {"launch_type": "local"}


def test_nested():
  set_bunch = {
    "resources.teacher.replica_hint": 10,
  }
  _ = dot_to_deps(set_bunch, "resources")
  assert _ == {"teacher": {"replica_hint": 10}}, "should contain nested dict"


def test_dot_prefix():
  _ = dot_to_deps({"resources.teacher.replicas_hint": 10}, "resources.teacher")
  assert _ == {"replicas_hint": 10}


def test_multile_prefix():
  _ = dot_to_deps({"resources.teacher.replicas_hint": 10}, "resources", "teacher")
  assert _ == {"replicas_hint": 10}


def test_mixed_prefixes():
  _ = dot_to_deps(
    {"root.resources.teacher.replicas_hint": 10}, "root.resources", "teacher"
  )
  assert _ == {"replicas_hint": 10}


def test_root_prefix():
  _ = dot_to_deps({"some": 10}, ".")
  assert _ == {"some": 10}


def test_root_empty_prefix():
  """to indicate root config, you need to pass in `.` instead."""
  import pytest

  with pytest.raises(AssertionError):
    _ = dot_to_deps({"some": 10}, "")


def test_outrageous_root_prefix():
  """full prefix string can not contain multiple consecutive dots."""
  import pytest

  with pytest.raises(AssertionError):
    _ = dot_to_deps({"some": 10}, "....")


def test_root_with_dependencies():
  """I don't like this. - Ge"""
  _ = dot_to_deps({"some": 10, "teacher.replicas_hints": 1}, ".")
  print(_)
