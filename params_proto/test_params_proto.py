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

    # args = proto(ParamsProto)
    G.npts = 10
    print(G)
    G.npts = 101
    print(G.npts)
    print(G)
    print("showing the dictionary from G: ", G.toDict())
    print("showing G.__proto__: ", G.__proto__)


def test_proto_signature():
    @cli_parse
    class G(ParamsProto):
        """some parameter proto"""
        npts: "number of points to sample from distribution" = 100

    @proto_signature(G.__proto__)
    def main_demo(**kwargs):
        print('npts = ', kwargs['npts'])
        pass

    # First way is to use proto_signature decorator. The dynamically generated signature
    # however does not show up in pyCharm. It does however, show during run time.
    import inspect
    print("main_demo<Function> signature:", inspect.signature(main_demo))
    main_demo(npts=10)
