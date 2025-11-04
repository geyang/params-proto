import pytest

from params_proto.v2.partial import proto_partial
from params_proto.v2.proto import ARGS, ParamsProto, Proto


@pytest.fixture
def clear_args():
  ARGS.clear()


def test_function_partial(clear_args):
  class G(ParamsProto):
    a = 23
    b = 29
    c = Proto(default=31, help="this is working")
    d = Proto(default=None, dtype=int, help="this is working")

  @proto_partial(G)
  def some_func(a, b, c, d, e="some_path"):
    assert a == 23, "the a entry should be 23."
    assert b == 29, "the a entry should be 29."
    assert c == 31, "the a entry should be 31."
    assert d is None, "the a entry should be None."
    assert e == "some_path", "use literal default"

  some_func()


def test_function_partial_with_keyword_only_arguments(clear_args):
  class G_2(ParamsProto):
    a = 23
    b = 29
    c = Proto(default=31, help="this is working")
    d = Proto(default=None, help="this is working")
    e = True

  # note: in this case a should not get
  #  the value from G. And E should not
  #  get the value from G either.
  @proto_partial(G_2)
  def some_func(a, *, b, c, d, e=None):
    assert a == 23, "the a entry should be 23."
    assert b == 29, "the a entry should be 29."
    assert c == 31, "the a entry should be 31."
    assert d is None, "the a entry should be None."
    assert e is None, "e should not get the value from G"

  e = None
  try:
    some_func()
  except Exception as _e:
    e = _e
  assert (
    str(e)
    == "test_function_partial_with_keyword_only_arguments.<locals>.some_func() missing 1 required positional argument: 'a'"
  )


def test_function_partial_dynamic_values(clear_args):
  class G_2(ParamsProto):
    a = 23
    b = 29
    c = Proto(default=31, help="this is working")
    d = Proto(default=None, help="this is working")
    e = True

  # note: in this case a should not get
  #  the value from G. And E should not
  #  get the value from G either.
  @proto_partial(G_2)
  def some_func(a, b, c, d, e=None):
    assert a == 2, "the a entry should be updated value"
    assert b == 4, "the a entry should be updated value."

  G_2.a = 2
  G_2.b = 4
  some_func()


def test_function_partial_override(clear_args):
  class G_2(ParamsProto):
    a = 23
    b = 29
    c = Proto(default=31, help="this is working")
    d = Proto(default=None, help="this is working")
    e = True

    @classmethod
    def __init__(cls, _deps=None):
      cls._update(_deps)
      cls.a = 2
      cls.b = 4

  # note: in this case a should not get
  #  the value from G. And E should not
  #  get the value from G either.
  @proto_partial(G_2)
  def some_func(a, b, c, d, e=None):
    assert a == 2, "the a entry should be updated value"
    assert b == 4, "the a entry should be updated value."
    assert c == 100, "this is being overridden"

  G_2()
  some_func(c=100)


def test_function_partial_class_method(clear_args):
  class G_2(ParamsProto):
    a = 23
    b = 29
    c = Proto(default=31, help="this is working")
    d = Proto(default=None, help="this is working")
    e = True

    @classmethod
    def __init__(cls, _deps=None):
      cls._update(_deps)
      cls.a = 2
      cls.b = 4

  # note: in this case a should not get
  #  the value from G. And E should not
  #  get the value from G either.
  class Yo:
    @proto_partial(G_2, method=True)
    def some_func(self, a, b, c, d, e=None):
      assert self.__class__ == Yo, "First arg is self."
      assert a == 2, "the a entry should be updated value"
      assert b == 4, "the a entry should be updated value."
      assert c == 100, "this is being overridden"

  G_2()
  yo = Yo()
  yo.some_func(c=100)
