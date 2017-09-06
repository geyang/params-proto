# params_proto, a collection of decorators that makes shell argument
passing declarative

## Todo

### Done
- [x] publish
- [x] add test

## Installation
```bash
pip install params_proto
```

## Usage

```python
from .params_proto import cli_parse, is_hidden, ParamsProto, proto_signature


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
```

## To Develop

```bash
git clone https://github.com/episodeyang/params_proto.git
cd params_proto
make dev
```

To test, run
```bash
make test
```

This `make dev` command should build the wheel and install it in your current python environment. Take a look at the [./Makefile](./Makefile) for details.

**To publish**, first update the version number, then do:
```bash
make publish
```
