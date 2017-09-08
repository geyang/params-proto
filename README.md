# params_proto, a collection of decorators that makes shell argument passing declarative


Now supports both python `3.52` as well as `3.6`!

## Todo

### Done
- [x] publish
- [x] add test
- [x] add `python3.52` test on top of `python3.6` test.

## Installation
```bash
pip install params_proto
```

## Usage

### To use a python namespace to declare commandline argments

**Note that this is python >= 3.6 only**. The reason being the annotation syntax
is only supported by python 3.6 an above. If you feel unhappy about this, just use
javascript.

For python <= 3.5, you can declare each default value as a tuple. `param_proto`
recognizes the python version that is being ran, and changes its behavior.

Below is a simple example how to use python namespace to declare command line
arguments:

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

    assert G.npts == 100
    G.npts = 10
    assert G.npts == 10
    assert vars(G) == {'npts': 10, 'num_epochs': 70000, 'num_tasks': 10, 'num_grad_steps': 1,
                       'num_points_sampled': 10, 'fix_amp': False}
    assert G._proto is not None, '_proto should exist'
```

### Setting Function Signatures using Python Namespace

sometimes, you have a function with wildcard keyword argument signature. It is
annoying to work with such functions because the static type analysis of the
IDE doesn't tell you what needs to go in.

I originally wrote this decorator to help with that case, however the dynamically
set function signature won't show up in the IDE in general. Use this for inspection
purposes if you like.

Below si the usage example and the test case:

```python
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
```

## To Develop

```bash
git clone https://github.com/episodeyang/params_proto.git
cd params_proto
make dev
```

To test, run the following under both python `3.52` and `3.6`.
```bash
make test
```

This `make dev` command should build the wheel and install it in your current python environment. Take a look at the [./Makefile](./Makefile) for details.

**To publish**, first update the version number, then do:
```bash
make publish
```
