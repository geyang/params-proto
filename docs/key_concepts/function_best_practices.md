# Function Parameter Best Practices

**--and How to Solve the Parameter Referencing Problem**

Earlier, we introduced params-proto using a simple function example. However, function-based parameter definitions have
a fundamental limitation: they break the linkage between parameter definitions and their usage. This happens because
functions create a new scope for local variables, disconnecting parameters from their original definitions.

Consider the following example:

```python
@params_proto.proto
def a_function(lr: float, batch_size: int = 32, optimizer: str = "adam"):
  """Train a model with the given parameters."""
  print(f"Learning Rate: {lr}")
  # ❌ No way to link 'lr' back to a centralized parameter definition
```

If this were a class-based python namespace, you could easily reference the parameter set as below:

```python
@proto
class Config:
  lr: float
  batch_size: int = 32
  optimizer: str = "adam"


# Now you can reference and override the defaults via:
for Config.lr in [0.01, 0.001, 0.0001]:
  print("current value:", Config.lr)
```

Although this works for the `Config` class, **vanilla Python functions do not provide an equivalent way to access and
modify parameter defaults!**

```{admonition} Why Can't I Use Functions to Define Hyper Parameters?
While Python does allow you to inspect a function's signature using `inspect.signature(fn)` or `fn.__signature__`,
modifying parameter defaults this way is cumbersome and doesn't provide a clean API for hyperparameter sweeps:

```python
import inspect

def fn(a: int = 0):
  print(f"We can refer to {a} inside the function, but not the default value.")

# You CAN access defaults via inspection:
sig = inspect.signature(fn)
print(sig.parameters['a'].default)  # 0

# But you CANNOT modify them cleanly:
# ❌ sig.parameters['a'].default = 10  # Parameters are read-only!
# ❌ fn.a = 10  # AttributeError: 'function' object has no attribute 'a'

# And you CANNOT iterate over them for sweeps:
# ❌ for fn.a in [0, 10, 20]:  # SyntaxError!
```

The function signature is read-only, and there's no natural way to write `for fn.a in [...]` like you can with class
attributes.

There are a few ways to work around this:

| Solution                        | Description                                                                                                           |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **Argument class**              | Pass a dataclass or class instance containing all parameters. Provides type safety and IDE support.                   |
| **`kwargs: Unpack`**            | (Python 3.11+) Use `Unpack[TypedDict]` to type hint `**kwargs`. Provides type checking while maintaining flexibility. |
| **Config singleton in closure** | Define parameters as class attributes and reference them directly. Enables iteration like `for Config.lr in [...]`.   |

## Under the Hood: How `@proto` Handles Functions

Before diving into the workarounds, it's worth understanding how params-proto solves this problem when you decorate a function with `@proto`.

When you write:

```python
@proto
def train(lr: float = 0.01, batch_size: int = 32):
  pass
```

params-proto doesn't just inspect the function—it **wraps** it in a special `ProtoWrapper` object. This wrapper serves a critical purpose: it provides the attribute access interface that vanilla Python functions lack.

### What ProtoWrapper Does

The `ProtoWrapper` intercepts attribute access and function calls to enable the same ergonomic API that classes provide:

```python
@proto
def train(lr: float = 0.01, batch_size: int = 32):
    print(f"Training with lr={lr}, batch_size={batch_size}")

# ProtoWrapper allows this:
train.lr = 0.001          # Store override
print(train.lr)           # Read current value → 0.001

# And enables sweeps like this:
for train.lr in [0.001, 0.01, 0.1]:
    train()  # Each call uses the updated lr value
```

Behind the scenes:
- **Parameter defaults** are extracted from the function signature and stored in `_defaults`
- **Overrides** are tracked in a separate `_overrides` dictionary
- **Attribute access** (`train.lr`) checks overrides first, then falls back to defaults
- **Function calls** merge defaults, overrides, and any kwargs before passing them to the original function

### Why Functions Need Wrapping (But Classes Don't)

Classes can use Python's metaclass protocol to intercept attribute access at the class level. The `@proto` decorator on classes creates them with a custom `ptype` metaclass that handles this transparently.

Functions, however, don't support metaclasses. To provide the same interface, we need an explicit wrapper object. The wrapper is completely transparent when you call the function—it behaves exactly like calling the original—but enables the attribute-based configuration API.

```{note}
When you use `@proto` on a function, `train` is not a function anymore—it's a `ProtoWrapper` instance. This is intentional and allows the ergonomic `train.lr = value` syntax to work.
```

### ProtoWrapper and Sweep Integration

The `ProtoWrapper` also provides special sweep mode support:

- **Normal mode**: Attribute assignments update `_overrides`
- **Sweep mode**: Attribute assignments are recorded for the sweep and trigger callbacks
- **Validation**: During sweeps, setting non-existent parameters raises an `AttributeError`

This integration allows seamless use of functions in hyperparameter sweeps alongside `@proto` classes:

```python
@proto.prefix
class Config:
    lr: float = 0.001

@proto.cli
def train(seed: int = 42):
    pass

# Both work seamlessly in sweeps:
with Sweep(Config, train).zip as sweep:
    Config.lr = [0.001, 0.01, 0.1]
    train.seed = [1, 2, 3]

# Generates 3 configs with both parameters
```

---

Let's look at each of the traditional workarounds in turn.

### Option 1: Argument Data Class

Dataclasses provide structured data with type hints that IDEs can understand and link to definitions.

```python
class TrainParams:
  """Parameters for the training function."""
  lr: float = 0.01
  batch_size: int = 32
  optimizer: str = 'adam'


def train(params: TrainParams) -> None:
  # ✓ Your IDE will link 'params.lr' back to the TrainParams definition
  print(f"Learning Rate: {params.lr}")
  print(f"Batch Size: {params.batch_size}")


# Usage:
config = TrainParams(lr=0.001)
train(params=config)
```

**Benefits:**

- Attribute access (`params.lr`) with IDE linkage
- Default values built-in
- Excellent IDE and static analyzer support
- Clear, centralized parameter definition

### Option 2: `fn(**kwargs: Unpack)` (Python 3.11+)

A more natural way is to use `Unpack[TypedDict]` to type hint `**kwargs`. This allows IDEs to understand the
structure of the dictionary and link back to the parameter definitions.

```{admonition} You cannot use a dataclass directly with (**kwargs)

This does NOT work:

     def train_with_dict(**config: TrainConfig) -> None:
       # Wrong: This means each VALUE must be of type TrainConfig
       pass

Instead, you must use `Unpack[TypedDict]` as shown below.
```

**✅ Using `Unpack[TypedDict]`:**

```python
from typing import TypedDict, Unpack


class TrainConfig(TypedDict):
  lr: float
  batch_size: int


def train_with_dict(**kwargs: Unpack[TrainConfig]) -> None:
  # ✓ The type checker knows config has 'lr' and 'batch_size' keys
  print(f"Learning Rate: {kwargs['lr']}")
  print(f"Batch Size: {kwargs['batch_size']}")


# Usage - You can unpack a dictionary:
my_config: TrainConfig = {'lr': 0.001, 'batch_size': 64}
train_with_dict(**my_config)

# Or call with explicit keyword arguments:
train_with_dict(lr=0.001, batch_size=64)
  ```

**Key Points:**

- **Runtime behavior:** The function receives a regular dictionary in the `config` variable
- **Static Analysis:** `Unpack[TrainConfig]` allows IDEs to validate that arguments match the keys and types defined in
  `TrainConfig`
- This restores the "link-back" functionality for parameter references

**Option 3: Config Singleton in Closure**

This is similar to Options 1 and 2 in that this also exposes a global configuration object. The main difference,
however, is that you no longer need to explicitly override the config values when calling the function. You can directly
apply overrides to the Config object, and the function will be referencing the updated value through the config
namespace during execution.

```python
from params_proto import proto


@proto
class Config:
  lr: float = 0.01
  batch_size: int = 32
  optimizer: str = "adam"


def train() -> None:
  # ✓ Direct access to global configuration
  print(f"Learning Rate: {Config.lr}")
  print(f"Batch Size: {Config.batch_size}")
  print(f"Optimizer: {Config.optimizer}")


# Usage - Direct attribute access and modification:
Config.lr = 0.001
train()

# Hyperparameter sweep:
for Config.lr in [0.01, 0.001, 0.0001]:
  print(f"Training with lr={Config.lr}")
  train()

```

**Benefits:**

- **Global accessibility:** Parameters can be accessed from anywhere in your code
- **Sweeps made easy:** Natural iteration over parameter values (`for Config.lr in [...]`)
- **Centralized configuration:** All parameters defined in one place
- **CLI generation:** Works seamlessly with params-proto's CLI generation features

**Trade-offs:**

- **Global state:** Parameters are mutable globals, which can make testing harder
- **Implicit dependencies:** Functions depend on external state rather than explicit arguments
- **Best for:** Scripts, experiments, and research code where sweep functionality is important

### Which Approach Should You Use?

Choose based on your use case:

- **Option 1 (Argument class):** Best for libraries, APIs, and production code where explicit dependencies are important
- **Option 2 (`kwargs: Unpack`):** Best when you need dictionary-style flexibility with type safety
- **Option 3 (Config singleton in closure):** Best for research code, experiments, and hyperparameter sweeps

**params-proto supports all three patterns**, allowing you to choose the right tool for your specific needs.
