"""v3 hyperparameter sweep implementation."""

import itertools
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from copy import deepcopy
from typing import (
  TYPE_CHECKING,
  Any,
  ContextManager,
  Dict,
  Iterable,
  List,
  Union,
  overload,
)

if TYPE_CHECKING:
  import pandas as pd

from ..proto import ProtoWrapper, ptype


class ParameterIterator:
  """
  Lazy iterator over parameter configurations.

  Supports operations like product (*), override (%), and power (**).
  """

  def __init__(self, iterator, protos=None):
    """
    Initialize parameter iterator.

    Args:
        iterator: Generator or iterable of config dicts
        protos: Optional dict of proto objects for context
    """
    self._iterator = iterator
    self._list = None
    self._protos = protos or {}

  def __iter__(self):
    """Iterate over configs, using cached list if available."""
    if self._list is not None:
      yield from self._list
    else:
      # Materialize and cache on first iteration
      self._list = list(self._iterator)
      yield from self._list

  @property
  def list(self):
    """Materialize all configs into a list."""
    if self._list is None:
      self._list = list(self._iterator)
    return self._list

  def to_list(self):
    """Materialize all configs into a list (alias for .list)."""
    return self.list

  def __mul__(self, other):
    """
    Cartesian product with another ParameterIterator.

    Example:
        piter1 = piter({Config.lr: [0.001, 0.01]})
        piter2 = piter({Config.batch_size: [32, 64]})
        combined = piter1 * piter2  # 4 combinations
    """
    if not isinstance(other, ParameterIterator):
      raise TypeError(f"Cannot multiply ParameterIterator with {type(other)}")

    # Materialize other to a list so we can iterate multiple times
    other_list = other.to_list()

    def product_iter():
      for config1 in self:
        for config2 in other_list:
          merged = {**config1, **config2}
          yield merged

    return ParameterIterator(product_iter())

  def __mod__(self, other):
    """
    Apply fixed overrides to all configs.

    Args:
        other: dict or ParameterIterator with override values

    Example:
        sweep = piter({Config.batch_size: [32, 64, 128]})
        with_lr = sweep % {"lr": 0.001}
        # or
        with_lr = sweep % piter({Config.lr: 0.001})
    """
    if isinstance(other, dict):
      overrides_iter = iter([other])
    elif isinstance(other, ParameterIterator):
      # Take first config from the piter as the override
      overrides_iter = iter(other)
    else:
      raise TypeError(f"Overrides must be dict or ParameterIterator, got {type(other)}")

    # Get the first (and typically only) override config
    try:
      overrides = next(overrides_iter)
    except StopIteration:
      raise ValueError("Override ParameterIterator is empty")

    def override_iter():
      for config in self:
        merged = {**config, **overrides}
        yield merged

    return ParameterIterator(override_iter())

  def __pow__(self, n):
    """
    Repeat each config n times.

    Useful for running multiple seeds/trials.

    Example:
        sweep = piter({Config.lr: [0.001, 0.01]})
        repeated = sweep ** 3  # Each config appears 3 times
    """
    if not isinstance(n, int) or n < 1:
      raise ValueError(f"Power must be a positive integer, got {n}")

    def repeat_iter():
      for config in self:
        for _ in range(n):
          yield config

    return ParameterIterator(repeat_iter())

  def __len__(self):
    """Return number of configs (materializes list)."""
    return len(self.list)


def piter(spec):
  """
  Create a parameter iterator from a specification dict.

  Args:
      spec: Dict mapping parameter names (strings) to values or lists of values.
            Keys should be parameter names, optionally with prefix (e.g., "config.lr").
            Values can be single values or lists/iterables.

  Returns:
      ParameterIterator that zips parameter lists element-wise.

  Example:
      # Element-wise zip (default behavior)
      piter({"lr": [0.001, 0.01], "batch_size": [32, 64]})
      # Creates 2 configs: (0.001, 32), (0.01, 64)

      # Fixed value
      piter({"seed": 200})
      # Creates 1 config with seed=200

      # For Cartesian product, use * operator:
      piter({"lr": [0.001, 0.01]}) * piter({"batch_size": [32, 64]})
      # Creates 4 configs (2 × 2)

      # With prefixes for multiple proto classes
      piter({"model.depth": [18, 50], "training.lr": [0.001, 0.01]})
      # Creates 2 configs (zipped)
  """
  # Convert values to lists if needed
  params = {}
  for key, values in spec.items():
    if not isinstance(key, str):
      raise TypeError(f"Parameter keys must be strings, got {type(key)}")
    param_values = values if isinstance(values, (list, tuple, range)) else [values]
    params[key] = param_values

  # Create iterator by zipping parameter lists
  def zip_iter():
    if not params:
      return

    keys = list(params.keys())
    value_lists = [params[k] for k in keys]

    # Zip parameter lists element-wise
    for combination in zip(*value_lists):
      config = dict(zip(keys, combination))
      yield config

  return ParameterIterator(zip_iter())


def dot_join(*keys):
  """Remove Nones from the keys, but not empty strings."""
  _ = [k for k in keys if k]
  if not _:
    return None
  return ".".join(_)


Item = namedtuple("Item", ["key", "value"])


def key_items(d, prefix=None):
  """
  Takes in tuples of [key, value], or [None, [[key, value], ...]].
  Yields items with optional prefix.
  """
  for k, vs in d:
    _ = dot_join(prefix, k)
    yield [Item(_, v) if _ else v for v in vs]


def flatten_items(row) -> Iterable[Item]:
  """Recursively flatten nested items."""
  if isinstance(row, Item):
    yield row
  elif isinstance(row, Iterable):
    for item in row:
      yield from flatten_items(item)
  else:
    yield row


class SweepProxy:
  """
  Proxy that switches between normal delegation and sweep interception.

  When not in sweep context: delegates all access directly to wrapped ptype class
  When in sweep context: intercepts setattr to record sweep configurations
  """

  def __init__(self, proto_obj):
    """
    Wrap a ptype class or ProtoWrapper.

    Args:
        proto_obj: The actual proto class (ptype instance) or ProtoWrapper
    """
    object.__setattr__(self, "_target", proto_obj)
    object.__setattr__(self, "_sweep_mode", False)
    object.__setattr__(self, "_sweep_data", {})
    object.__setattr__(self, "_sweep_callback", None)

  def _enable_sweep_mode(self, callback):
    """Enter sweep mode with a callback for recording values."""
    object.__setattr__(self, "_sweep_mode", True)
    object.__setattr__(self, "_sweep_callback", callback)
    object.__setattr__(self, "_sweep_data", {})

  def _disable_sweep_mode(self):
    """Exit sweep mode and return to normal delegation."""
    object.__setattr__(self, "_sweep_mode", False)
    object.__setattr__(self, "_sweep_callback", None)
    object.__setattr__(self, "_sweep_data", {})

  @property
  def _prefix(self):
    """Return prefix for the proto object (kebab-case for v3)."""
    target = object.__getattribute__(self, "_target")

    # Check if it's a metaclass-based proto class
    if isinstance(target, type) and isinstance(target, ptype):
      return type.__getattribute__(target, "__proto_prefix__")
    elif isinstance(target, ProtoWrapper):
      return target._prefix  # Delegate to ProtoWrapper's _prefix property
    return None

  def _update(self, __d: Dict[str, Any] = None, **kwargs):
    """Update overrides from dict or kwargs."""
    target = object.__getattribute__(self, "_target")

    # Helper to get annotations
    def get_annotations():
      if isinstance(target, type) and isinstance(target, ptype):
        return type.__getattribute__(target, "__proto_annotations__")
      elif isinstance(target, ProtoWrapper):
        return target._params
      return {}

    # Helper to set override
    def set_override(key, value):
      if isinstance(target, type) and isinstance(target, ptype):
        overrides = type.__getattribute__(target, "__proto_overrides__")
        overrides[key] = value
      elif isinstance(target, ProtoWrapper):
        target._overrides[key] = value

    annotations = get_annotations()

    if __d:
      prefix = self._prefix
      if prefix:
        prefix_key = f"{prefix}."
        for k, v in __d.items():
          if k.startswith(prefix_key):
            param_name = k[len(prefix_key) :]
            if param_name in annotations:
              set_override(param_name, v)
          elif "." not in k:
            if k in annotations:
              set_override(k, v)
      else:
        for k, v in __d.items():
          if "." not in k:
            if k in annotations:
              set_override(k, v)

    for k, v in kwargs.items():
      if k in annotations:
        set_override(k, v)

  def __setattr__(self, name, value):
    """Intercept attribute setting based on mode."""
    if name.startswith("_"):
      object.__setattr__(self, name, value)
      return

    # Check if we're in sweep mode
    sweep_mode = object.__getattribute__(self, "_sweep_mode")

    if sweep_mode:
      # In sweep mode: record the value and call callback
      sweep_data = object.__getattribute__(self, "_sweep_data")
      sweep_data[name] = {"_value": value}

      callback = object.__getattribute__(self, "_sweep_callback")
      if callback:
        prefix = self._prefix
        callback(name, value, prefix)
    else:
      # Normal mode: delegate directly to target
      target = object.__getattribute__(self, "_target")
      setattr(target, name, value)

  def __getattribute__(self, name):
    """Delegate attribute access based on mode."""
    # Handle internal attributes
    if name.startswith("_") or name in ("_prefix", "_enable_sweep_mode",
                                         "_disable_sweep_mode", "_update"):
      return object.__getattribute__(self, name)

    # Check if we're in sweep mode and have recorded data
    try:
      sweep_mode = object.__getattribute__(self, "_sweep_mode")
      if sweep_mode:
        sweep_data = object.__getattribute__(self, "_sweep_data")
        if name in sweep_data:
          return sweep_data[name]["_value"]
    except AttributeError:
      pass

    # Delegate to target
    target = object.__getattribute__(self, "_target")
    return getattr(target, name)

  def __dir__(self):
    """Return parameters only, not internals."""
    target = object.__getattribute__(self, "_target")
    if isinstance(target, type) and isinstance(target, ptype):
      annotations = type.__getattribute__(target, "__proto_annotations__")
      return list(annotations.keys())
    elif isinstance(target, ProtoWrapper):
      return list(target._params.keys())
    return []

  def __getstate__(self):
    """Support for pickling/deepcopy."""
    target = object.__getattribute__(self, "_target")
    return {
      "_target": target,
      "_sweep_mode": False,  # Don't copy sweep mode
      "_sweep_data": {},
      "_sweep_callback": None,
    }

  def __setstate__(self, state):
    """Support for unpickling/deepcopy."""
    object.__setattr__(self, "_target", state["_target"])
    object.__setattr__(self, "_sweep_mode", state.get("_sweep_mode", False))
    object.__setattr__(self, "_sweep_data", state.get("_sweep_data", {}))
    object.__setattr__(self, "_sweep_callback", state.get("_sweep_callback", None))


class Sweep:
  """
  Hyperparameter sweep for v3 @proto decorated classes and functions.

  Supports product, zip, chain, and set operations for combining parameter configurations.
  """

  def __init__(self, *protos):
    """
    Initialize Sweep with proto objects.

    Args:
        *protos: ProtoWrapper instances or metaclass-based proto classes
    """
    # Store proto objects directly (both ptype and ProtoWrapper can switch modes themselves)
    self.root: Dict[Union[str, object], Any] = {}
    for p in protos:
      # Get prefix for keying
      if isinstance(p, type) and isinstance(p, ptype):
        prefix = type.__getattribute__(p, "__proto_prefix__")
        key = prefix or p
      elif isinstance(p, ProtoWrapper):
        # ProtoWrapper now has built-in sweep mode support
        key = p._prefix or p
      else:
        key = getattr(p, "_prefix", None) or p

      self.root[key] = p

    self.stack = [[]]
    self._d = None
    self.__original = None
    self.__each_fn = None

  def __len__(self):
    return len(self.list)

  def __getitem__(self, item: Union[slice, int]):
    """Support indexing and slicing of sweep configurations."""
    if isinstance(item, slice):
      assert item.step != 0, "step cannot be zero."
      if (
        (item.start and item.start < 0)
        or (item.stop and item.stop < 0)
        or (item.step and item.step < 0)
      ):
        for override in self.list[item]:
          for org, proto in zip(self.original, self.noot.values(), strict=False):
            proto._update(**org)
            proto._update(override)
          yield override
        return
      for i, el in enumerate(self):
        if item.start is not None and i < item.start:
          continue
        if item.step is None or (i - (item.start or 0)) % item.step == 0:
          yield el
        if item.stop is None:
          continue
        if i >= item.stop - 1:
          break
    elif isinstance(item, int):
      if item < 0:
        for override in self.list[item]:
          for org, proto in zip(self.original, self.noot.values(), strict=False):
            proto._update(**org)
            proto._update(override)
          yield override
      for i, el in enumerate(self):
        if i == item:
          for org, proto in zip(self.original, self.noot.values(), strict=False):
            proto._update(**org)
            proto._update(el)
          yield el
          break
    else:
      raise NotImplementedError(f"slicing is not implemented for {item}")

  def items(self):
    """Return enumerated sweep configurations."""
    return enumerate(self)

  @property
  def list(self):
    """Convert sweep to list of configuration dicts."""
    return [*iter(self)]

  def to_list(self):
    """Convert sweep to list of configuration dicts (alias for .list)."""
    return self.list

  def __mul__(self, other):
    """
    Cartesian product of two sweeps or a sweep and a ParameterIterator.

    Returns an iterator over all combinations from both sweeps.

    Example:
        sweep1 = Sweep(Config1).product
        Config1.lr = [0.001, 0.01]

        sweep2 = Sweep(Config2).product
        Config2.batch_size = [32, 64]

        combined = sweep1 * sweep2
        # Yields 4 configs: all combinations of lr and batch_size
    """
    if not isinstance(other, (Sweep, ParameterIterator)):
      raise TypeError(f"Cannot multiply Sweep with {type(other)}")

    # Materialize other if it's a ParameterIterator to allow multiple iterations
    other_list = other.to_list() if isinstance(other, ParameterIterator) else None

    def product_iter():
      for config1 in self:
        iter_other = other_list if other_list is not None else other
        for config2 in iter_other:
          # Merge the two configs
          merged = {**config1, **config2}
          yield merged

    return ParameterIterator(product_iter())

  def __mod__(self, other):
    """
    Apply fixed parameter overrides to all configs in the sweep.

    Args:
        other: dict or ParameterIterator with override values

    Returns an iterator where each config has the overrides merged in.

    Example:
        sweep = Sweep(Config).product
        Config.batch_size = [32, 64, 128]

        with_fixed_lr = sweep % {"lr": 0.001}
        # or
        with_fixed_lr = sweep % piter({"lr": 0.001})
        # All configs will have lr=0.001 with varying batch_size
    """
    if isinstance(other, dict):
      overrides_iter = iter([other])
    elif isinstance(other, ParameterIterator):
      # Take first config from the piter as the override
      overrides_iter = iter(other)
    else:
      raise TypeError(f"Overrides must be dict or ParameterIterator, got {type(other)}")

    # Get the first (and typically only) override config
    try:
      overrides = next(overrides_iter)
    except StopIteration:
      raise ValueError("Override ParameterIterator is empty")

    def override_iter():
      for config in self:
        # Merge overrides into config
        merged = {**config, **overrides}
        yield merged

    return ParameterIterator(override_iter())

  def __pow__(self, n):
    """
    Repeat each config n times.

    Useful for running multiple seeds/trials.

    Example:
        sweep = Sweep(Config).product
        Config.seed = [10, 20]
        repeated = sweep ** 3  # Each config appears 3 times (total 6)
    """
    if not isinstance(n, int) or n < 1:
      raise ValueError(f"Power must be a positive integer, got {n}")

    def repeat_iter():
      for config in self:
        for _ in range(n):
          yield config

    return ParameterIterator(repeat_iter())

  @property
  def dataframe(self):
    """Convert sweep to pandas DataFrame."""
    import pandas as pd

    return pd.DataFrame(self.list)

  @property
  def __dict__(self):
    """Return dict mapping parameter names to lists of values."""
    if self._d:
      return self._d
    self._d = defaultdict(list)
    for config in self:
      for k, v in config.items():
        self._d[k].append(v)
    return self._d

  @property
  def noot(self):
    """Return deep copy of root protos."""
    return deepcopy(self.root)

  @property
  def snack(self):
    """Return deep copy of stack."""
    return deepcopy(self.stack)

  def __enter__(self):
    """Enter context for sweep.set mode."""
    self.stack.append([])

    for proto in self.root.values():
      # Create a callback that records sweep parameters
      def make_callback(prefix):
        def callback(name, value, p=prefix):
          self.set_param(name, [value], prefix=p)
        return callback

      # Get prefix - works for ptype, ProtoWrapper, and SweepProxy
      if isinstance(proto, type) and isinstance(proto, ptype):
        prefix = type.__getattribute__(proto, "__proto_prefix__")
      elif isinstance(proto, ProtoWrapper):
        prefix = proto._prefix
      else:
        prefix = proto._prefix

      proto._enable_sweep_mode(make_callback(prefix))

    return self

  def __exit__(self, *args):
    """Exit context and process sweep stack."""
    for proto in self.root.values():
      proto._disable_sweep_mode()

    frame = self.stack.pop(-1)
    result = itertools.product(*key_items(frame))
    self.set_param(None, result)

  @property
  def original(self):
    """Get original parameter values for each proto."""
    if self.__original is None:
      self.__original = []
      for proto in self.noot.values():
        original_vals = {}
        # Get current parameter values
        for param_name in dir(proto):
          if not param_name.startswith("_"):
            try:
              original_vals[param_name] = getattr(proto, param_name)
            except AttributeError:
              pass
        self.__original.append(original_vals)
    return self.__original

  def __iter__(self):
    """Iterate over all sweep configurations."""
    for row in itertools.chain(*[it.value for it in self.snack[-1]]):
      override = dict(flatten_items(row))
      for org, proto in zip(self.original, self.noot.values(), strict=False):
        proto._update(**org)
        # Only apply relevant overrides
        # Get prefix - works for ptype, ProtoWrapper, and SweepProxy
        if isinstance(proto, type) and isinstance(proto, ptype):
          prefix = type.__getattribute__(proto, "__proto_prefix__")
        elif isinstance(proto, ProtoWrapper):
          prefix = proto._prefix
        else:
          prefix = proto._prefix

        if prefix:
          filtered = {k: v for k, v in override.items() if k.startswith(f"{prefix}.")}
          proto._update(filtered)
        else:
          proto._update({k: v for k, v in override.items() if "." not in k})

      if callable(self.__each_fn):
        with Sweep(*self.noot.values()) as sweep:
          self.__each_fn(*self.noot.values())
        for deps in sweep:
          yield {k: v for k, v in itertools.chain(override.items(), deps.items())}
      else:
        yield override

  def set_param(self, name, params, prefix=None):
    """Add parameter to current sweep stack."""
    item = Item(dot_join(prefix, name), params)
    self.stack[-1].append(item)

  @property
  @contextmanager
  def product(self) -> ContextManager[None]:
    """Context manager for Cartesian product of parameters."""
    self.stack.append([])
    try:
      for proto in self.root.values():
        # Get prefix - works for ptype, ProtoWrapper, and SweepProxy
        if isinstance(proto, type) and isinstance(proto, ptype):
          prefix = type.__getattribute__(proto, "__proto_prefix__")
        elif isinstance(proto, ProtoWrapper):
          prefix = proto._prefix
        else:
          prefix = proto._prefix

        proto._enable_sweep_mode(lambda name, value, p=prefix: self.set_param(name, value, prefix=p))
      yield self
    finally:
      for proto in self.root.values():
        proto._disable_sweep_mode()

      frame = self.stack.pop(-1)
      result = itertools.product(*key_items(frame))
      self.set_param(None, result)

  @property
  @contextmanager
  def zip(self) -> ContextManager[None]:
    """Context manager for element-wise zip of parameters."""
    self.stack.append([])
    try:
      for proto in self.root.values():
        # Get prefix - works for ptype, ProtoWrapper, and SweepProxy
        if isinstance(proto, type) and isinstance(proto, ptype):
          prefix = type.__getattribute__(proto, "__proto_prefix__")
        elif isinstance(proto, ProtoWrapper):
          prefix = proto._prefix
        else:
          prefix = proto._prefix

        proto._enable_sweep_mode(lambda name, value, p=prefix: self.set_param(name, value, prefix=p))
      yield self
    finally:
      for proto in self.root.values():
        proto._disable_sweep_mode()

      frame = self.stack.pop(-1)
      result = list(zip(*key_items(frame), strict=False))
      self.set_param(None, result)

  @property
  @contextmanager
  def set(self) -> ContextManager[None]:
    """Context manager for setting fixed parameter values."""
    try:
      yield self.__enter__()
    finally:
      self.__exit__()

  @property
  @contextmanager
  def chain(self) -> ContextManager[None]:
    """Context manager for chaining multiple sweep configurations."""
    self.stack.append([])
    try:
      for proto in self.root.values():
        # Get prefix - works for ptype, ProtoWrapper, and SweepProxy
        if isinstance(proto, type) and isinstance(proto, ptype):
          prefix = type.__getattribute__(proto, "__proto_prefix__")
        elif isinstance(proto, ProtoWrapper):
          prefix = proto._prefix
        else:
          prefix = proto._prefix

        proto._enable_sweep_mode(lambda name, value, p=prefix: self.set_param(name, value, prefix=p))
      yield self
    finally:
      for proto in self.root.values():
        proto._disable_sweep_mode()

      frame = self.stack.pop(-1)
      result = itertools.chain(*(value for k, value in frame))
      self.set_param(None, result)

  def each(self, fn):
    """
    Register a function to be called for each configuration.

    Useful for setting values that dynamically depend on other sweep values.
    """
    self.__each_fn = fn
    return self

  @overload
  def save(self, filename: str = "sweep.jsonl", overwrite: bool = True, verbose: bool = True) -> None:
    """Save sweep configurations to JSONL file."""
    ...

  @overload
  def save(self, filename: "os.PathLike[str]", overwrite: bool = True, verbose: bool = True) -> None:
    """Save sweep configurations to JSONL file."""
    ...

  def save(self, filename="sweep.jsonl", overwrite=True, verbose=True):
    """
    Save sweep configurations to JSONL file.

    Args:
        filename: Path to output file (str or PathLike)
        overwrite: If True, overwrite existing file; if False, append
        verbose: If True, print save confirmation
    """
    import json
    import os
    from urllib import parse

    # Convert Path objects to string
    filename_str = os.fspath(filename) if hasattr(os, 'fspath') else str(filename)

    with open(filename_str, "w" if overwrite else "a+") as f:
      for item in self.list:
        f.write(json.dumps(item) + "\n")

    if verbose:
      try:
        from termcolor import colored as c

        print(
          c("saved", "blue"),
          c(len(self.list), "green"),
          c("items to", "blue"),
          filename_str,
          ".",
          "file://" + parse.quote(os.path.realpath(filename_str)),
        )
      except ImportError:
        print(f"Saved {len(self.list)} items to {filename_str}")

  @staticmethod
  def log(deps, filename):
    """Append single config to JSONL file."""
    import json

    with open(filename, "a+") as f:
      f.write(json.dumps(deps) + "\n")

  @staticmethod
  def read(filename):
    """Read JSONL file into list of dicts."""
    import json

    sweep = []
    with open(filename, "r") as f:
      line = f.readline().strip()
      while line:
        if not line.startswith("//"):
          sweep.append(json.loads(line.strip()))
        line = f.readline().strip()
    return sweep

  @overload
  def load(self, file: str, strict: bool = True, silent: bool = False) -> "Sweep":
    """Load sweep configurations from JSONL file."""
    ...

  @overload
  def load(self, file: "os.PathLike[str]", strict: bool = True, silent: bool = False) -> "Sweep":
    """Load sweep configurations from JSONL file."""
    ...

  @overload
  def load(
    self, deps: List[Dict[str, Any]], strict: bool = True, silent: bool = False
  ) -> "Sweep":
    """Load sweep configurations from list of dicts."""
    ...

  @overload
  def load(
    self, df: "pd.DataFrame", strict: bool = True, silent: bool = False
  ) -> "Sweep":
    """Load sweep configurations from pandas DataFrame."""
    ...

  def load(self, file: Union[str, "os.PathLike[str]", List[Dict[str, Any]], "pd.DataFrame"] = "sweep.jsonl", strict: bool = True, silent: bool = False):
    """
    Load sweep configurations from JSONL file or list.

    Args:
        file: Filename (str or PathLike), list of dicts, or pandas DataFrame
        strict: Raise error on missing attributes (default True)
        silent: Suppress warnings about missing attributes (default False)

    Returns:
        self for chaining
    """
    import os
    import pandas as pd

    self.file = file

    # Convert to DataFrame
    if isinstance(file, (str, os.PathLike)):
      # Convert Path to string
      file_str = os.fspath(file) if hasattr(os, 'fspath') else str(file)
      deps = self.read(file_str)
    elif isinstance(file, list):
      deps = file
    else:
      deps = None

    df = pd.DataFrame(deps) if deps is not None else file

    if not isinstance(df, pd.DataFrame):
      raise TypeError(f"{type(file)} is not supported")

    with self.zip:
      for full_key in df:
        prefix, *keys = full_key.split(".")
        if prefix in self.root:
          proto = self.root[prefix]
          param_name = ".".join(keys) if keys else None
          if param_name and not hasattr(proto, keys[0]):
            if strict:
              raise KeyError(f'{proto} does not contain the key "{full_key}"')
            if not silent:
              print(f'{proto} does not contain the key "{full_key}"')
            # Skip setting if attribute doesn't exist and strict=False
            continue
          if param_name:
            setattr(proto, param_name, df[full_key].values.tolist())
        else:
          for k, proto in self.root.items():
            if isinstance(k, str):
              continue
            if hasattr(proto, prefix):
              setattr(proto, full_key, df[full_key].values.tolist())
              break
          else:
            if strict:
              raise KeyError(
                f'The key "{full_key}" does not appear in any of the Arguments'
              )
            if not silent:
              print(f'The key "{full_key}" does not appear in any of the Arguments')
    return self
