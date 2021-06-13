import itertools
from collections import namedtuple, defaultdict
from contextlib import contextmanager
from typing import TypeVar, ContextManager, Iterable, Union, Dict

from params_proto.neo_proto import Meta, ParamsProto


def dot_join(*keys):
    """remove Nones from the keys, but not '', """
    _ = [k for k in keys if k]
    if not _:
        return None
    return ".".join(_)


Item = namedtuple('Item', ['key', 'value'])


def key_items(d, prefix=None):
    """
    Takes in tuples of [key, value], or [None, [[key, value], ...]]
    returns wtf
    """
    for k, vs in d:
        _ = dot_join(prefix, k)
        yield [Item(_, v) if _ else v for v in vs]


def flatten_items(row) -> Iterable[Item]:
    if isinstance(row, Item):
        yield row
    elif isinstance(row, Iterable):
        for item in row:
            yield from flatten_items(item)
    else:
        yield row


T = TypeVar('ParamsProto')


class Sweep:
    _d = None
    __original = None
    __noot = None

    __each_fn = None

    def each(self, fn):
        self.__each_fn = fn

    # noinspection PyProtectedMember
    def __init__(self, *protos: Meta):
        # the ParamsProto is updatable via proto._update(dot_dict)
        # use object itself as key if _prefix is missing
        self.root: Dict[str, ParamsProto] = {p._prefix or p: p for p in protos}
        self.stack = [[]]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item: Union[slice, int, float]):
        if isinstance(item, slice):
            assert item.step != 0, "step can not be zero."
            if (item.start and item.start < 0) or (item.stop and item.stop < 0) or (item.step and item.step < 0):
                for override in self.list[item]:
                    for org, proto in zip(self.original, self.noot.values()):
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
            # need-test: Not tested from a quick glance.
            if item < 0:
                for override in self.list[item]:
                    for org, proto in zip(self.original, self.noot.values()):
                        proto._update(**org)
                        proto._update(override)
                    yield override
            for i, el in enumerate(self):
                if i == item:
                    for org, proto in zip(self.original, self.noot.values()):
                        proto._update(**org)
                        proto._update(el)
                    yield el
                    break
        else:
            raise NotImplementedError(f"slicing is not implemented for {item}")

    def items(self):
        return enumerate(self)

    @property
    def list(self):
        """returns self as a list. Currently not idempotent. Might become idempotent in the future."""
        return [*iter(self)]

    @property
    def __dict__(self):
        if self._d:
            return self._d
        self._d = defaultdict(list)
        for config in self:
            for k, v in config.items():
                self._d[k].append(v)
        return self._d

    @property
    def noot(self):
        from copy import deepcopy
        return deepcopy(self.root)

    @property
    def snack(self):
        from copy import deepcopy
        return deepcopy(self.stack)

    def __enter__(self):
        self.stack.append([])
        for proto in self.root.values():
            set_hook = lambda _, k, v, p=proto._prefix: self.set_param(k, [v], prefix=p)
            proto._add_hook(set_hook)

        return self

    def __exit__(self, *args):
        for proto in self.root.values():
            proto._pop_hooks()

        frame = self.stack.pop(-1)
        result = itertools.product(*key_items(frame))
        self.set_param(None, result)

    @property
    def original(self):

        if self.__original is None:
            self.__original = []
            for proto in self.noot.values():
                # noinspection PyCallByClass
                def no_reset(k):
                    return getattr(type.__getattribute__(proto, k), "accumulant", False)

                self.__original.append({k: v
                                        for k, v in vars(proto).items()
                                        if not no_reset(k)})
        return self.__original

    def __iter__(self):
        for row in itertools.chain(*[it.value for it in self.snack[-1]]):
            override = dict(flatten_items(row))
            for org, proto in zip(self.original, self.noot.values()):
                proto._update(**org)
                # only apply those key-value pairs that appear in the original.
                proto._update(override if proto._prefix else {k: v for k, v in override.items() if k in org})

            if callable(self.__each_fn):
                with Sweep(*self.noot.values()) as sweep:
                    self.__each_fn(*self.noot.values())
                for deps in sweep:
                    yield {k: v for k, v in itertools.chain(override.items(), deps.items())}
            else:
                yield override

    def set_param(self, name, params, prefix=None):
        item = Item(dot_join(prefix, name), params)
        self.stack[-1].append(item)

    @property
    @contextmanager
    def product(self) -> ContextManager[None]:
        self.stack.append([])
        try:
            for proto in self.root.values():
                prefix = proto._prefix
                proto._add_hook(lambda _, *args, p=prefix: self.set_param(*args, prefix=p))
            yield self
        finally:
            for proto in self.root.values():
                proto._pop_hooks()

            frame = self.stack.pop(-1)
            result = itertools.product(*key_items(frame))
            self.set_param(None, result)

    @property
    @contextmanager
    def zip(self) -> ContextManager[T]:
        self.stack.append([])
        try:
            for proto in self.root.values():
                prefix = proto._prefix
                proto._add_hook(lambda _, *args, p=prefix: self.set_param(*args, prefix=p))
            yield self
        finally:
            for proto in self.root.values():
                proto._pop_hooks()

            frame = self.stack.pop(-1)
            result = list(zip(*key_items(frame)))
            self.set_param(None, result)

    @property
    @contextmanager
    def set(self) -> ContextManager[T]:
        try:
            yield self.__enter__()
        finally:
            self.__exit__()

    @property
    @contextmanager
    def chain(self) -> ContextManager[T]:
        self.stack.append([])
        try:
            for proto in self.root.values():
                prefix = proto._prefix
                proto._add_hook(lambda _, *args, p=prefix: self.set_param(*args, prefix=p))
            yield self
        finally:
            for proto in self.root.values():
                proto._pop_hooks()

            frame = self.stack.pop(-1)
            result = itertools.chain(*(value for k, value in frame))
            self.set_param(None, result)

    def save(self, filename="sweep.jsonl", overwrite=True, verbose=True):
        import json
        from termcolor import colored as c
        # todo: connect to ml-logger to setup managed sweep
        with open(filename, 'w' if overwrite else 'a+') as f:
            for item in self.list:
                f.write(json.dumps(item) + '\n')

        if verbose:
            print(
                c("saved", "blue"),
                c(len(self.list), "green"),
                c("items to", "blue"),
                filename,
            )

    def load(self, filename="sweep.jsonl", strict=True, silent=False):
        """
        Loading sweep state from a jsonl file:

        Note: **Important Caveat** When multiple prefix-free ParamsProto objects are present,
            We sweep through all of the proto objects and sets the attribute to the first
            proto with the correct key. This first-attr approach works because the ParamsProto
            object also generates argparse parameters, which means repetitive arguments are
            not possible.
            ~
            However this would fail in cases where attributes are dynamically added to an
            argument object. The `sweep.jsonl` file loses this type of information, therefore
            there is no way to recover this type of attributes. So the user should try to
            either use PrefixProto, or explicitly define the attributes.

        Usage Pattern:

            sweep = Sweep(Args, RUN).load('sweep.jsonl')
            for i, deps in enumerate(sweep):
                assert RUN.job_id == i + 1, "the job_id in that sweep.json should be 1-based."

        """
        import json
        import pandas as pd
        from termcolor import colored
        sweep = []
        with open(filename, 'r') as f:
            l = f.readline().strip()
            while l:
                # need to handle end of line
                if not l.startswith("#"):
                    sweep.append(json.loads(l.strip()))
                l = f.readline().strip()

        # print(*sweep, sep="\n")
        df = pd.DataFrame(sweep)

        with self.zip:
            for full_key in df:
                first, *keys = full_key.split('.')
                if first in self.root:
                    setattr(self.root[first], '.'.join(keys), df[full_key].values.tolist())
                else:
                    for k, proto in self.root.items():
                        if isinstance(k, str):
                            continue
                        if hasattr(proto, first):
                            setattr(proto, full_key, df[full_key].values.tolist())
                            break
                    else:
                        msg = colored(f'The key "', "red") + \
                              colored(f'{full_key}', "green") + \
                              colored(f'" ', "red") + \
                              colored(f'does not appear in any of the Arguments', "red")
                        if strict:
                            raise KeyError(msg)
                        if not silent:
                            print(msg)
        return self
