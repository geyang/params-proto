import sys
from pprint import pprint
from textwrap import dedent

from params_proto import is_hidden, cli_parse, Proto, ParamsProto, proto_signature, BoolFlag


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
    assert 'decorated' in globals()


def test_is_hidden():
    # test is_hidden
    assert is_hidden('_test_name')
    assert not is_hidden('test_name')


def test_cli_proto_simple():
    @cli_parse
    class G(ParamsProto):
        """Supervised MAML in tensorflow"""
        npts = 100

    assert G.npts == 100


def test_cli_proto():
    @cli_parse
    class G(ParamsProto):
        """Supervised MAML in tensorflow"""
        npts = Proto(100, help="number of points to sample from distribution")
        num_epochs = Proto(70000, help="number of epochs to train")
        num_tasks = Proto(10, help="number of tasks in the inner loop")
        num_grad_steps = Proto(1, help="number of gradient descent steps in the inner loop")
        num_points_sampled = Proto(10, help="effectively the k-shot")
        eval_grad_steps = Proto([0, 1, 10], help="the grad steps evaluated with full sample")
        fix_amp = Proto(False, help="controls the sampling, fix the amplitude of the sample distribution if True")
        render = BoolFlag(False, help="turn on the rendering")
        no_dump = BoolFlag(True, help="turn off the data dump. By default dump when no flag is present.")

    assert G.npts == 100
    G.npts = 10
    assert G.npts == 10
    assert vars(G) == {'npts': 10, 'num_epochs': 70000, 'num_tasks': 10, 'num_grad_steps': 1,
                       'num_points_sampled': 10, 'fix_amp': False,
                       'eval_grad_steps': [0, 1, 10],
                       "render": False, "no_dump": True}
    assert G._proto is not None, '_proto should exist'

    @cli_parse
    class G(ParamsProto):
        """some parameter proto"""
        n = 1
        npts = Proto(100, help="number of points to sample from distribution")
        ok = True

    @proto_signature(G._proto)
    def main_demo(**kwargs):
        print('npts = ', kwargs['npts'])
        return kwargs['npts']

    # First way is to use proto_signature decorator. The dynamically generated signature
    # however does not show up in pyCharm. It does however, show during run time.
    import inspect

    assert main_demo(npts=10) == 10
    print("main_demo<Function> signature:", inspect.signature(main_demo))
    assert str(inspect.signature(main_demo)) == "(n=1, npts=100, ok=True)"


def test_proto_to_dict():
    @cli_parse
    class G(ParamsProto):
        """some parameter proto"""
        npts: "number of points to sample from distribution" = 100

    assert vars(G) == {'npts': 100}


from textwrap import dedent
from subprocess import check_call, CalledProcessError, check_output


def test_from_command_line():
    """this is not used in the actual testing."""
    script = dedent("""
        # python -c "import os; print(os.getcwd())"
        echo "\n----------- .py:test_from_command_line() ---------------"
        echo "\n=================== std.out output ====================="
        echo "\n>>>>>>>>>>>> do this in parent directory <<<<<<<<<<<<<<<"
        pwd
        source activate plan2vec &&
        pip install -e . &&
        python ./params_proto/test_fixtures/main.py -h
        """)
    check_call(script, shell=True)


if __name__ == "__main__":
    test_decorator()
    test_is_hidden()
    test_cli_proto()
    test_cli_proto_simple()
    test_proto_to_dict()
