"""v3 hyperparameter sweep implementation."""
import itertools
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, ContextManager, Dict, Iterable, Union

from ..proto import ProtoClass, ProtoWrapper, ptype
from .proxies import ClassProxy, FuncProxy, PrefixProxy


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


class ProtoProxy:
    """
    Proxy wrapper that provides Sweep interface for v3 ProtoClass/ProtoWrapper.

    This keeps the main ProtoClass clean while providing the hooks and methods
    needed for Sweep integration.
    """

    def __init__(self, proto_obj):
        """
        Wrap a v3 ProtoClass, ProtoWrapper, or metaclass-based proto class.

        Args:
            proto_obj: A ProtoClass, ProtoWrapper instance, or metaclass-based class
        """
        object.__setattr__(self, "_proto", proto_obj)
        object.__setattr__(self, "_set_hooks", [])
        object.__setattr__(self, "_get_hooks", [])

    @property
    def _prefix(self):
        """Return prefix for the proto object (lowercase for v3)."""
        proto = object.__getattribute__(self, "_proto")

        # Check if it's a metaclass-based proto class
        if isinstance(proto, type) and isinstance(proto, ptype):
            return type.__getattribute__(proto, "__proto_prefix__")
        # Check if it's a ProtoClass wrapper (old style)
        elif isinstance(proto, ProtoClass):
            return proto._cls.__name__.lower() if proto._is_prefix else None
        elif isinstance(proto, ProtoWrapper):
            return proto._name.lower() if proto._is_prefix else None
        return None

    def _add_hooks(self, set_hook, get_hook=None):
        """Add setter/getter hooks for Sweep integration."""
        proto = object.__getattribute__(self, "_proto")

        # Check if it's a metaclass-based proto class
        if isinstance(proto, type) and isinstance(proto, ptype):
            # Access __proto_sweep_hooks__ via type.__getattribute__ to bypass metaclass
            hooks = type.__getattribute__(proto, "__proto_sweep_hooks__")
            hooks.append(set_hook)
        # Old wrapper-based approach (for backward compatibility during transition)
        elif hasattr(proto, "_sweep_hooks"):
            proto._sweep_hooks.append(set_hook)

    def _pop_hooks(self):
        """Remove last hook."""
        proto = object.__getattribute__(self, "_proto")

        # Check if it's a metaclass-based proto class
        if isinstance(proto, type) and isinstance(proto, ptype):
            hooks = type.__getattribute__(proto, "__proto_sweep_hooks__")
            if hooks:
                hooks.pop()
        # Old wrapper-based approach
        elif hasattr(proto, "_sweep_hooks"):
            if proto._sweep_hooks:
                proto._sweep_hooks.pop()

    def _update(self, __d: Dict[str, Any] = None, **kwargs):
        """Update overrides from dict or kwargs."""
        proto = object.__getattribute__(self, "_proto")

        # Helper to get annotations
        def get_annotations():
            if isinstance(proto, type) and isinstance(proto, ptype):
                return type.__getattribute__(proto, "__proto_annotations__")
            elif isinstance(proto, ProtoClass):
                return proto._annotations
            elif isinstance(proto, ProtoWrapper):
                return proto._params
            return {}

        # Helper to set override
        def set_override(key, value):
            if isinstance(proto, type) and isinstance(proto, ptype):
                overrides = type.__getattribute__(proto, "__proto_overrides__")
                overrides[key] = value
            elif isinstance(proto, ProtoClass):
                proto._overrides[key] = value
            elif isinstance(proto, ProtoWrapper):
                proto._overrides[key] = value

        annotations = get_annotations()

        if __d:
            prefix = self._prefix
            if prefix:
                prefix_key = f"{prefix}."
                for k, v in __d.items():
                    if k.startswith(prefix_key):
                        param_name = k[len(prefix_key):]
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

    def __getattr__(self, name):
        """Delegate attribute access to wrapped proto object, with hook support."""
        # Get hooks using object.__getattribute__ to avoid recursion
        try:
            get_hooks = object.__getattribute__(self, "_get_hooks")
            if get_hooks:
                for hook in get_hooks:
                    if hook:
                        result = hook(self, name)
                        if result is not None:
                            # Handle Proto wrapper (has .value attribute)
                            return result.value if hasattr(result, "value") else result
        except AttributeError:
            pass

        # Delegate to wrapped proto
        proto = object.__getattribute__(self, "_proto")
        return getattr(proto, name)

    def __setattr__(self, name, value):
        """Handle attribute setting with hook support."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            # Call hooks if present
            try:
                set_hooks = object.__getattribute__(self, "_set_hooks")
                if set_hooks:
                    for hook in set_hooks:
                        result = hook(self, name, value)
                        if result is not None:
                            return result
            except AttributeError:
                pass

            # Delegate to wrapped proto
            proto = object.__getattribute__(self, "_proto")
            setattr(proto, name, value)

    def __dir__(self):
        """Return parameters only, not internals."""
        proto = object.__getattribute__(self, "_proto")
        if isinstance(proto, type) and isinstance(proto, ptype):
            annotations = type.__getattribute__(proto, "__proto_annotations__")
            return list(annotations.keys())
        elif isinstance(proto, ProtoClass):
            return list(proto._annotations.keys())
        elif isinstance(proto, ProtoWrapper):
            return list(proto._params.keys())
        return []

    def __getstate__(self):
        """Support for pickling/deepcopy - serialize the wrapped proto object."""
        proto = object.__getattribute__(self, "_proto")
        return {
            "_proto": proto,
            "_set_hooks": [],  # Don't copy hooks
            "_get_hooks": [],  # Don't copy hooks
        }

    def __setstate__(self, state):
        """Support for unpickling/deepcopy - restore the wrapped proto object."""
        object.__setattr__(self, "_proto", state["_proto"])
        object.__setattr__(self, "_set_hooks", state.get("_set_hooks", []))
        object.__setattr__(self, "_get_hooks", state.get("_get_hooks", []))


class Sweep:
    """
    Hyperparameter sweep for v3 @proto decorated classes and functions.

    Supports product, zip, chain, and set operations for combining parameter configurations.
    """

    def __init__(self, *protos):
        """
        Initialize Sweep with proto objects.

        Args:
            *protos: ProtoClass, ProtoWrapper instances, or metaclass-based proto classes
        """
        # Wrap v3 proto objects in ProtoProxy
        self.root: Dict[Union[str, object], ProtoProxy] = {}
        for p in protos:
            # Check if it's a metaclass-based proto class
            if isinstance(p, type) and isinstance(p, ptype):
                proxy = ProtoProxy(p)
                key = proxy._prefix or proxy
                self.root[key] = proxy
            elif isinstance(p, (ProtoClass, ProtoWrapper)):
                proxy = ProtoProxy(p)
                # Use prefix as key if available, otherwise use the proxy itself
                key = proxy._prefix or proxy
                self.root[key] = proxy
            else:
                # Already a proxy or compatible object
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
            data = {}

            def set_hook(_, k, v, p=proto._prefix):
                # Wrap value to distinguish from None
                data[k] = {"_value": v}
                return self.set_param(k, [v], prefix=p)

            def get_hook(_, k, p=proto._prefix):
                return data.get(k, None)

            proto._add_hooks(set_hook, get_hook)

        return self

    def __exit__(self, *args):
        """Exit context and process sweep stack."""
        for proto in self.root.values():
            proto._pop_hooks()

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
                prefix = proto._prefix
                proto._add_hooks(lambda _, *args, p=prefix: self.set_param(*args, prefix=p))
            yield self
        finally:
            for proto in self.root.values():
                proto._pop_hooks()

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
                prefix = proto._prefix
                proto._add_hooks(lambda _, *args, p=prefix: self.set_param(*args, prefix=p))
            yield self
        finally:
            for proto in self.root.values():
                proto._pop_hooks()

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
                prefix = proto._prefix
                proto._add_hooks(lambda _, *args, p=prefix: self.set_param(*args, prefix=p))
            yield self
        finally:
            for proto in self.root.values():
                proto._pop_hooks()

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

    def save(self, filename="sweep.jsonl", overwrite=True, verbose=True):
        """Save sweep configurations to JSONL file."""
        import json
        import os
        from urllib import parse

        with open(filename, "w" if overwrite else "a+") as f:
            for item in self.list:
                f.write(json.dumps(item) + "\n")

        if verbose:
            try:
                from termcolor import colored as c
                print(
                    c("saved", "blue"),
                    c(len(self.list), "green"),
                    c("items to", "blue"),
                    filename,
                    ".",
                    "file://" + parse.quote(os.path.realpath(filename)),
                )
            except ImportError:
                print(f"Saved {len(self.list)} items to {filename}")

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

    def load(self, file="sweep.jsonl", strict=True, silent=False):
        """
        Load sweep configurations from JSONL file or list.

        Args:
            file: Filename (str), list of dicts, or pandas DataFrame
            strict: Raise error on missing attributes (default True)
            silent: Suppress warnings about missing attributes (default False)

        Returns:
            self for chaining
        """
        import pandas as pd

        self.file = file

        if isinstance(file, str):
            file = self.read(file)
        if isinstance(file, list):
            df = pd.DataFrame(file)
        elif isinstance(file, pd.DataFrame):
            df = file
        else:
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
