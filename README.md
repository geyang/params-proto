# `params-proto`, A Python Decorator That Gives Your Model Parameters Super-power

- 2019/12/09: Just finished my DeepMind Internship. Now params_proto contain
    a new proto implementation, and a complementary hyperparameter search
    library! See `neo_proto` and `neo_hyper`.
- 2019/06/11: Now supports `tab-completion` at the command line!
- 2018/11/08: Now supports both python `3.52` as well as `3.6`! :bangbang::star:

## What is "Experiment Parameter Hell"?

"Experiemnt Parameter Hell" occurs when you have more than twenty parameters for your ML project that are all defined as string/function parameters with `click` or `argparse`. Sometimes these parameters are defined in a launch script and passes through five layers of function calls during an experiment.

<img width="60%" align="right" alt="autocompletion demo" src="./figures/params-proto-autocompletion.gif"></img>

Your Python IDEs work very hard on static code analysis to intelligently make you more productive, and the "parameter hell" breaks all of that.

### Step 1: Declarative Pattern to the Rescue!

For this reason, you want to avoid using dictionaries or opaque `argparse` definitions as much as possible. Instead, you want to write those declaratively, so that your IDE can actually help you navigate through those layers of function calls. The hyper-parameter library, `params_proto` makes this easy, by integrating python namespaces (a bare python class) with `argparse`,  so that on the python side you get auto-completion, and from the command line you can pass in changes.

**Installation**

First let's install `params-proto` and its supporting module `waterbear`

```bash
pip install params-proto waterbear
```

Then to declare your hyperparameters, you can write the following in a `your_project/soft_ac/config.py` file:

```python
import sys
from params_proto.neo_proto import ParamsProto, Flag, Proto, PrefixProto

# this is the first config schema
class Args(PrefixProto):
    """Soft-actor Critic Implementation with SOTA Performance
    """

    debug = True if "pydevd" in sys.modules else False

    cuda = Flag("cuda tend to be slower.")
    seed = 42
    env_name = "FetchReach-v1"
    n_workers = 1 if debug else 12
    v_lr = 1e-3
    pi_lr = 1e-3
    n_initial_rollouts = 0 if debug else 100
    n_test_rollouts = 15
    demo_length = 20
    clip_inputs = Flag()
    normalize_inputs = Flag()

# this is the second schema
class LfGR(PrefixProto):
    # reporting
    use_lfgr = True
    start = 0 if Args.debug else 10
    store_interval = 10
    visualization_interval = 10
```

### Step 2: Sweeping Hyper-parameters 🔥

Then you an sweep the hyperparameter via the following declarative pattern:

```python
from rl import main, Args
from params_proto.neo_hyper import Sweep

if __name__ == '__main__':
    from lp_analysis import instr

    with Sweep(Args, LfGR) as sweep:
	# override the default
        Args.pi_lr = 3e-3
        Args.clip_inputs = True # this was a flag
        
	# override the second config object
	LfGR.visualization_interval = 40

	# product between the zipped and the seed
        with sweep.product:
	    # similar to python zip, unpacks a list of values.
            with sweep.zip:
                Args.env_name = ['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1', 'FetchSlide-v1']
                Args.n_epochs = [4, 12, 12, 20]
                Args.n_workers = [5, 150, 200, 500]
	    # the seed is sweeped at last
            Args.seed = [100, 200, 300, 400, 500, 600]

    for i, deps in sweep.items():
        thunk = instr(main, deps, _job_postfix=f"{Args.env_name}")
        print(deps)
```

and it should print out a list of dictionaries that looks like:

```bash
{Args.pi_lr: 3e-3, Args.clip_inputs: True, LfGR.visualization_interval: 40, Args.env_name: "FetchReach-v1", ... Args.seed: 100}
{Args.pi_lr: 3e-3, Args.clip_inputs: True, LfGR.visualization_interval: 40, Args.env_name: "FetchReach-v1", ... Args.seed: 200}
{Args.pi_lr: 3e-3, Args.clip_inputs: True, LfGR.visualization_interval: 40, Args.env_name: "FetchReach-v1", ... Args.seed: 300}
...
```

<img width="60%" align="right" alt="spec_files" src="figures/spec_files.png"></img>
## Where Can I find Documentation?

Look at the specification file at [./test_params_proto/*.py](test_params_proto) , which is part of the integrated test. These scripts contains the most comprehensive set of usage patters!!

The new version has a `neo_` prefix. We will deprecate the older (non-neo) version in a few month.


## Writing documentation as uhm..., man page?

<img width="60%" align="right" alt="man page" src="./figures/man-page.png"></img>

`Params-Proto` exposes your argument namespace's doc string as the usage note. For users of your code, there is no better help than the one that comes with the script itself!

> With `params-proto`, your help is only one `-h` away :)

And **Your code becomes the documentation.**

## Tab-completion for your script!

`params_proto` uses `argparse` together with `argcomplete`, which enables command line autocomplete on tabs! To enable run

```python
pip install params-proto
# then:
activate-global-python-argcomplete
```

For details, see [`argcomplete`'s documentation](https://github.com/kislyuk/argcomplete#installation).


## Why Use Params_Proto Instead of Click or Argparse?

Because this declarative, singleton pattern allows you to:

> Place all of the arguments under a namespace that can be statically checked.

so that your IDE can:

1. Find usage of each argument
2. jump from *anywhere* in your code base to the declaration of that argument
3. refactor your argument name **in the entire code base** automatically

`Params_proto` is the declarative way to write command line arguments, and is the way to go for ML projects.


## How to override when calling from python

It is very easy to over-ride the parameters when you call your function: have most of your training code **directly** reference the parser namespace (your configuration namespace really), and just monkey patch the attribute.

`params-proto` works very well with the clound ML launch tool [jaynes](https://github.com/episodeyang/jaynes). Take a look at the automagic awesomeness of [jaynes](https://github.com/episodeyang/jaynes):)


-------------------

## Old Usage

## Simple Example (with batteries included!!):battery:

```python
# this.code.py
from params_proto import cli_parse, BoolFlag, Proto

@cli_parse
class Args:
    """
    [README]
        Generator for the 2D Particle Map Dataset. See Usage help below:
    """
    load = Proto(None, dtype=str, help="to visualize existing data located at this path")
    x_dim = Proto(2, help="The dimension for the observation space")
    data_size = Proto(20, help="The size of the dataset. Note we x2 because we generate transitions.")
    show_plot = BoolFlag(True, help="Shows the plot when true.")

def train():
    D = Discriminator(Args.x_dim)

def launch(**kwargs):
    Args.update(kwargs)

if __name__ == "__main__":
    launch(show_plot=True)
```


now, if you run this code, it gives you this help in the command line:
```python
(/Users/ge/anaconda/envs/some-project) ➜ git:(master) python -m this.code.py -h
usage: generate.py [-h] [--load LOAD] [--x-dim X_DIM] [--data-size DATA_SIZE]
                   [--show-plot]

[README] Generator for the 2D Particle Map Dataset. See Usage help below:

optional arguments:
  -h, --help            show this help message and exit
  --load LOAD           to visualize existing data located at this path
  --x-dim X_DIM         The dimension for the observation space
  --data-size DATA_SIZE
                        The size of the dataset. Note we x2 because we
                        generate transitions.
  --show-plot           Shows the plot when true.
```

Now, isn't this awesome? :bang::stars:

### To use a python namespace to declare commandline argments

**Updated** We now use a `Proto` helper function to declare the argparse arguments! See example below:

## Simple example showing how to use python namespace to declare command line arguments:

**note**: for boolean, use `bool` or `"bool"`. `params_proto` will automatically use `distutils.util.strtobool` to parse it into `bool`. Details look [here](https://docs.python.org/2/distutils/apiref.html?highlight=distutils.util#distutils.util.strtobool)

```python
from .params_proto import cli_parse, is_hidden, Proto, ParamsProto, proto_signature


def test_cli_proto():
    @cli_parse
    class G(ParamsProto):
        """Supervised MAML in tensorflow"""
        npts = Proto(100, help="number of points to sample from distribution")
        num_epochs = Proto(70000, help="number of epochs to train")
        num_tasks = Proto(10, help="number of tasks in the inner loop")
        num_grad_steps = Proto(1, help="number of gradient descent steps in the inner loop")
        num_points_sampled = Proto(10, help="effectively the k-shot")
        eval_grad_steps = Proto([0, 1, 10], type=bool, help="the grad steps evaluated with full sample")
        fix_amp = Proto(False, help="controls the sampling, fix the amplitude of the sample distribution if True")

    assert G.npts == 100
    G.npts = 10
    assert G.npts == 10
    assert vars(G) == {'npts': 10, 'num_epochs': 70000, 'num_tasks': 10, 'num_grad_steps': 1,
                       'num_points_sampled': 10, 'fix_amp': False,
                       'eval_grad_steps': [0, 1, 10]}
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
