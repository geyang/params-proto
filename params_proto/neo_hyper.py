import itertools
from collections import namedtuple, defaultdict
from contextlib import contextmanager
from functools import partial
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
    __list = None

    # noinspection PyProtectedMember
    def __init__(self, *protos: Meta):
        # the ParamsProto is updatable via proto._update(dot_dict)
        self.root: Dict[str, ParamsProto] = {p._prefix: p for p in protos}
        self.stack = [[]]

    def __getitem__(self, item: Union[slice, int, float]):
        if isinstance(item, slice):
            assert item.step != 0, "step can not be zero."
            if (item.start and item.start < 0) or (item.stop and item.stop < 0) or (item.step and item.step < 0):
                for override in self.list[item]:
                    for proto in self.root.values():
                        proto._update(override)
                    yield override
            for i, el in enumerate(self):
                if item.start is not None and i < item.start:
                    continue
                if item.step is None or (i - (item.start or 0)) % item.step == 0:
                    for proto in self.root.values():
                        proto._update(el)
                    yield el
                if item.stop is None:
                    continue
                if i >= item.stop - 1:
                    break
        elif isinstance(item, int):
            if item < 0:
                for override in self.list[item]:
                    for proto in self.root.values():
                        proto._update(override)
                    yield override
            for i, el in enumerate(self):
                if i == item:
                    for proto in self.root.values():
                        proto._update(el)
                    yield el
                    break
        else:
            raise NotImplementedError(f"slicing is not implemented for {item}")

    @property
    def list(self):
        """returns self as a list."""
        if self.__list:
            return self.__list
        self.__list = list(self)
        return self.__list

    @property
    def __dict__(self):
        if self._d:
            return self._d
        self._d = defaultdict(list)
        for config in self:
            for k, v in config.items():
                self._d[k].append(v)
        return self._d

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

    def __iter__(self):
        for row in itertools.chain(*[it.value for it in self.stack[-1]]):
            override = dict(flatten_items(row))

            for proto in self.root.values():
                proto._update(override)

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
            yield None
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
            yield None
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
            yield None
        finally:
            for proto in self.root.values():
                proto._pop_hooks()

            frame = self.stack.pop(-1)
            result = itertools.chain(*(value for k, value in frame))
            self.set_param(None, result)
