import itertools
from collections import namedtuple
from contextlib import contextmanager
from typing import TypeVar, ContextManager, Iterable

from params_proto.neo_proto import is_private


def dot_join(*keys):
    """remove Nones from the keys, but not '', """
    _ = [k for k in keys if k is not None]
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


# support nested query.
class ProxyObject(object):
    def __init__(self, hook, prefix=None):
        self.__prefix = prefix
        self.__hook = hook

    @property
    def __dict__(self):
        return {k: v for k, v in super().__dict__.items() if not is_private(k)}

    def __setattr__(self, key, value):
        if key.startswith("_ProxyObject_"):
            return object.__setattr__(self, key, value)

        self.__hook(dot_join(self.__prefix, key), value)

    def __getattr__(self, item):
        if item.startswith("_ProxyObject_"):
            return object.__getattribute__(self, item)

        return ProxyObject(self.__hook, prefix=dot_join(self.__prefix, item))


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
    def __init__(self):
        self.stack = [[]]  # root stack frame

    def __iter__(self):
        root_stack = self.stack[-1]
        key, rows = root_stack[0]
        assert key is None, "this is the root stack"
        for row in rows:
            a = flatten_items(row)
            yield dict(a)

    def set_param(self, name, params, prefix=None):
        item = Item(dot_join(prefix, name), params)
        self.stack[-1].append(item)

    @contextmanager
    def product(self, Strut: T) -> ContextManager[T]:
        self.stack.append([])
        proxy = ProxyObject(self.set_param, prefix=Strut._prefix)
        try:
            yield proxy
        finally:
            frame = self.stack.pop(-1)
            result = itertools.product(*key_items(frame))
            self.set_param(None, result)

    @contextmanager
    def zip(self, Strut: T) -> ContextManager[T]:
        self.stack.append([])
        proxy = ProxyObject(self.set_param, prefix=Strut._prefix)
        try:
            yield proxy
        finally:
            frame = self.stack.pop(-1)
            result = zip(*key_items(frame))
            self.set_param(None, result)

    @contextmanager
    def set(self, Strut: T) -> ContextManager[T]:
        self.stack.append([])
        _ = lambda k, v: self.set_param(k, [v])
        proxy = ProxyObject(_, prefix=Strut._prefix)
        try:
            yield proxy
        finally:
            frame = self.stack.pop(-1)
            result = itertools.product(*key_items(frame))
            self.set_param(None, result)


sweep = Sweep()
