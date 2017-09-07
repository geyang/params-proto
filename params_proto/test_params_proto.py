from .params_proto import cli_parse, is_hidden, ParamsProto, proto_signature


def test_is_hidden():
    # test is_hidden
    assert is_hidden('_test_name')
    assert not is_hidden('test_name')


def test_cli_proto():
    @cli_parse
    class G(ParamsProto):
        """Supervised MAML in tensorflow"""
        npts: "number of points to sample from distribution" = 100
        num_epochs: "number of epochs to train" = 70000
        num_tasks: "number of tasks in the inner loop" = 10
        num_grad_steps: "number of gradient descent steps in the inner loop" = 1
        num_points_sampled: "effectively the k-shot" = 10
        fix_amp: "controls the sampling, fix the amplitude of the sample distribution if True" = False

    assert G.npts == 100
    G.npts = 10
    assert G.npts == 10
    assert vars(G) == {'npts': 10, 'num_epochs': 70000, 'num_tasks': 10, 'num_grad_steps': 1,
                       'num_points_sampled': 10, 'fix_amp': False}
    assert G._proto is not None, '_proto should exist'


def test_proto_signature():
    @cli_parse
    class G(ParamsProto):
        """some parameter proto"""
        npts: "number of points to sample from distribution" = 100

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
    @cli_parse
    class G(ParamsProto):
        """some parameter proto"""
        npts: "number of points to sample from distribution" = 100

    assert vars(G) == {'npts': 100}
