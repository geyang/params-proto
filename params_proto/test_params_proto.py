import sys

from .params_proto import is_hidden, cli_parse, ParamsProto, proto_signature


def test_decorator():
    script = """
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
"""
    exec(script, globals())
    print("")
    assert a == 4
    assert b == 44
    assert 'decorated' in globals()


def test_is_hidden():
    # test is_hidden
    assert is_hidden('_test_name')
    assert not is_hidden('test_name')


def test_cli_proto_simple():
    @cli_parse
    class G(ParamsProto):
        """Supervised MAML in tensorflow"""
        npts = 100,

    if sys.version_info >= (3, 6):
        assert G.npts == (100,)
    else:
        assert G.npts == 100


def test_cli_proto():
    if sys.version_info >= (3, 6):
        script = """
@cli_parse
class G(ParamsProto):
    \"\"\"Supervised MAML in tensorflow\"\"\"
    npts: "number of points to sample from distribution" = 100
    num_epochs: "number of epochs to train" = 70000
    num_tasks: "number of tasks in the inner loop" = 10
    num_grad_steps: "number of gradient descent steps in the inner loop" = 1
    num_points_sampled: "effectively the k-shot" = 10
    fix_amp: "controls the sampling, fix the amplitude of the sample distribution if True" = False
        """
    else:
        script = """
@cli_parse
class G(ParamsProto):
    \"\"\"Supervised MAML in tensorflow\"\"\"
    npts = 100, "number of points to sample from distribution"
    num_epochs = 70000, "number of epochs to train"
    num_tasks = 10, "number of tasks in the inner loop"
    num_grad_steps = 1, "number of gradient descent steps in the inner loop"
    num_points_sampled = 10, "effectively the k-shot"
    fix_amp = False, "controls the sampling, fix the amplitude of the sample distribution if True"
        """
    exec(script, globals())

    assert G.npts == 100
    G.npts = 10
    assert G.npts == 10
    assert vars(G) == {'npts': 10, 'num_epochs': 70000, 'num_tasks': 10, 'num_grad_steps': 1,
                       'num_points_sampled': 10, 'fix_amp': False}
    assert G._proto is not None, '_proto should exist'


def test_proto_signature():
    if sys.version_info >= (3, 6):
        script = "@cli_parse\n" \
                 "class G(ParamsProto):\n" \
                 "    \"\"\"some parameter proto\"\"\"\n" \
                 "    npts: \"number of points to sample from distribution\" = 100\n"
    else:
        script = "@cli_parse\n" \
                 "class G(ParamsProto):\n" \
                 "    \"\"\"some parameter proto\"\"\"\n" \
                 "    npts = 100, \"number of points to sample from distribution\"\n"
    exec(script, globals())

    @proto_signature(G._proto)
    def main_demo(**kwargs):
        print('npts = ', kwargs['npts'])
        return kwargs['npts']

    # First way is to use proto_signature decorator. The dynamically generated signature
    # however does not show up in pyCharm. It does however, show during run time.
    import inspect

    assert main_demo(npts=10) == 10
    print("main_demo<Function> signature:", inspect.signature(main_demo))
    assert str(inspect.signature(main_demo)) == "(npts=100)"


def test_proto_to_dict():
    if sys.version_info >= (3, 6):
        script = "@cli_parse\n" \
                 "class G(ParamsProto):\n" \
                 "    \"\"\"some parameter proto\"\"\"\n" \
                 "    npts: \"number of points to sample from distribution\" = 100\n"
    else:
        script = "@cli_parse\n" \
                 "class G(ParamsProto):\n" \
                 "    \"\"\"some parameter proto\"\"\"\n" \
                 "    npts = 100, \"number of points to sample from distribution\"\n"
    exec(script, globals())

    assert vars(G) == {'npts': 100}
