from params_proto.v1 import (
  BoolFlag,
  ParamsProto,
  Proto,
  cli_parse,
  is_hidden,
  params_proto,
  prefix_proto,
  proto_partial,
  proto_signature,
)


def test_decorator():
  script = dedent("""
    def decorate(fn):
        def wrapper(x):
            return fn(x + 10)
        return wrapper;
            
    def original(x):
        return x * 4
        
    @decorate
    def decorated(x):
        return x * 4
    a = original(1)
    b = decorated(1)
    """)
  exec(script, globals())
  print("")
  assert a == 4
  assert b == 44
  assert "decorated" in globals()


def test_is_hidden():
  # test is_hidden
  assert is_hidden("_test_name")
  assert not is_hidden("test_name")


def test_cli_proto_simple():
  params_proto.PARSER = None

  @cli_parse
  class G(ParamsProto):
    """Supervised MAML in tensorflow"""

    npts = 100

  assert G.npts == 100


def test_cli_proto():
  params_proto.parser = None

  @cli_parse
  class G(ParamsProto):
    """Supervised MAML in tensorflow"""

    npts = Proto(100, help="number of points to sample from distribution")
    num_epochs = Proto(70000, help="number of epochs to train")
    num_tasks = Proto(10, help="number of tasks in the inner loop")
    num_grad_steps = Proto(1, help="number of gradient descent steps in the inner loop")
    num_points_sampled = Proto(10, help="effectively the k-shot")
    eval_grad_steps = Proto(
      [0, 1, 10], help="the grad steps evaluated with full sample"
    )
    fix_amp = Proto(
      False,
      help="controls the sampling, fix the amplitude of the sample distribution if True",
    )
    render = BoolFlag(False, help="turn on the rendering")
    no_dump = BoolFlag(
      True, help="turn off the data dump. By default dump when no flag is present."
    )

  assert G.npts == 100
  G.npts = 10
  assert G.npts == 10
  assert vars(G) == {
    "npts": 10,
    "num_epochs": 70000,
    "num_tasks": 10,
    "num_grad_steps": 1,
    "num_points_sampled": 10,
    "fix_amp": False,
    "eval_grad_steps": [0, 1, 10],
    "render": False,
    "no_dump": True,
  }
  assert G._proto is not None, "_proto should exist"

  params_proto.parser = None

  @cli_parse
  class G(ParamsProto):
    """some parameter proto"""

    n = 1
    npts = Proto(100, help="number of points to sample from distribution")
    ok = True

  @proto_signature(G._proto)
  def main_demo(**kwargs):
    print("npts = ", kwargs["npts"])
    return kwargs["npts"]

  # First way is to use proto_signature decorator. The dynamically generated signature
  # however does not show up in pyCharm. It does however, show during run time.
  import inspect

  assert main_demo(npts=10) == 10
  print("main_demo<Function> signature:", inspect.signature(main_demo))
  assert str(inspect.signature(main_demo)) == "(n=1, npts=100, ok=True)"


def test_proto_to_dict():
  params_proto.parser = None

  @cli_parse
  class G(ParamsProto):
    """some parameter proto"""

    npts: "number of points to sample from distribution" = 100

  assert vars(G) == {"npts": 100}


from textwrap import dedent


def test_from_command_line():
  """this is not used in the actual testing."""

  @cli_parse
  class G:
    some_arg = Proto(0, aliases=["-s"])

  print(G.__parser.format_help())
  assert (
    G.__parser.format_help()
    == dedent(""" 
        usage: _jb_pytest_runner.py [-h] [--some-arg SOME_ARG]
        
        options:
          -h, --help            show this help message and exit
          --some-arg SOME_ARG, -s SOME_ARG
                                N/A
        """).lstrip()
  )


def test_function_partial():
  params_proto.parser = None

  @prefix_proto
  class G:
    a = 23
    b = 29
    c = Proto(default=31, help="this is working")
    d = Proto(default=None, help="this is working")

  @proto_partial(G)
  def some_func(a, b, c, d, e="some_path"):
    assert a == 23, "the a entry should be 23."
    assert b == 29, "the a entry should be 29."
    assert c == 31, "the a entry should be 31."
    assert d is None, "the a entry should be None."
    assert e == "some_path", "use literal default"

  some_func()


def test_function_partial_with_keyword_only_arguments():
  params_proto.PARSER = None

  @prefix_proto
  class G_2:
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


if __name__ == "__main__":
  test_decorator()
  test_is_hidden()
  test_cli_proto()
  test_cli_proto_simple()
  test_proto_to_dict()
  test_function_partial()
  test_function_partial_with_keyword_only_arguments()
