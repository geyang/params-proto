# `params-proto`, Modern Hyper Parameter Management for Machine Learning

- 2022/07/04: 
    * Move `neo_proto` to top-level, move older `params_proto` to `v1` namespace.
    * Implement nested update via [global prefix](https://github.com/geyang/params_proto/blob/master/test_params_proto/test_neo_proto.py#L278). No relative update via `**kwargs`, yet
    * Fix `to_value` bug in Flag
- 2021/06/16: Proto now supports using environment variables as default.
- 2021/06/13: 5 month into my postdoc at MIT, add `sweep.save("sweep.jsonl")` to dump
    the sweep into a `jsonl` file for large scale experiments on AWS.
- 2019/12/09: Just finished my DeepMind Internship. Now `params-proto` contain
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
from params_proto.proto import ParamsProto, Flag, Proto, PrefixProto


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

### Step 2: Sweeping Hyper-parameters :fire:

Then you an sweep the hyperparameter via the following declarative pattern:

```python
from rl import main, Args
from params_proto.hyper import Sweep

if __name__ == '__main__':
    from lp_analysis import instr

    with Sweep(Args, LfGR) as sweep:
        # override the default
        Args.pi_lr = 3e-3
        Args.clip_inputs = True  # this was a flag

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

    # You can save the sweep into a `jsonl` file
    sweep.save('sweep.jsonl')

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

## To Develop And Contribute

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




