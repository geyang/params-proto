import argparse
import inspect
import re
import sys
from typing import TypeVar

from waterbear import DefaultBear


def is_hidden(k: str) -> bool:
    """return True is method is hidden"""
    return bool(re.match("^_.*$", k))


def props_to_dict(obj):
    """
    takes class or instance object and returns the attributes that are not hidden.

    :param obj:
    :return:

    Usage Example

        Python 3.6.2 | packaged by conda-forge | (default, Jul 23 2017, 23:01:38)
        [GCC 4.2.1 Compatible Apple LLVM 6.1.0 (clang-602.0.53)] on darwin

    Take this namespace for example:

    ```python
    class Test():
        name = 10
        key = 100
    ```

    if you instantiate it, you see nothing.

    ```python
    t = Test()
    vars(t)  # {}
    t.__dict__  # {}
    ```

    but if you just use it as namespace, you can get the attributes:
    ```python
    t = Test
    vars(t)  # {"name": 10, "key": 100}
    ```

    Attributes normally assigned to self inside `__init__` function shows up anyways.
    ```python
    class Test():
        def __init__(self):
            self.name = 10
            self.key = 100

    t = Test()
    vars(t)  # {'name': 10, 'key': 100}
    t.__dict__  # {'name': 10, 'key': 100}
    ```
    """
    return {k: v for k, v in vars(obj).items() if not is_hidden(k)}


T = TypeVar('T')


class ParamsProto(DefaultBear):
    """Parameter Prototype Class, has toDict method and __proto__ attribute for the original namespace object."""

    def __init__(self, proto, **d):
        super().__init__(None, **d)
        self._proto = proto

    @property
    def __dict__(self):
        return {k: v for k, v in super().__dict__.items() if not is_hidden(k)}


# noinspection PyTypeChecker
def cli_parse(proto: T) -> T:
    """parser command line options, and repackage into a typed object.
    :type proto: T
    """
    parser = argparse.ArgumentParser(description=T.__doc__)
    for k, v in proto.__dict__.items():
        if is_hidden(k):
            continue
        k_normalized = k.replace('_', '-')
        if sys.version_info >= (3, 6):
            default = v
            try:
                help_str = proto.__annotations__[k]
            except (KeyError, AttributeError):  # todo: use proper python logging for debug
                help_str = "N/A"
        else:  # use array as proto attribute value for python <= 3.5
            assert len(v) >= 1, "for python version <= 3.5, use a tuple to define the parameter prototype."
            default, *_ = v
            if len(_) > 0:
                help_str = _[0]
            else:
                help_str = "N/A"
        data_type = type(default)
        parser.add_argument('--{k}'.format(k=k_normalized), default=default, type=data_type, help=help_str)

    if sys.version_info <= (3, 6):
        params = ParamsProto(proto, **{k: v[0] for k, v in vars(proto).items() if not is_hidden(k)})
    else:
        params = ParamsProto(proto, **{k: v for k, v in vars(proto).items() if not is_hidden(k)})

    args, unknow_args = parser.parse_known_args()
    params.update(vars(args))

    return params


# noinspection PyUnusedLocal
def proto_signature(parameter_prototype, need_self=False):
    def decorate(f):
        if need_self is True:
            print('ha')
        # Need to have return type as well.
        __doc__ = f.__doc__

        if sys.version_info <= (3, 6):
            s_str = "{k}=parameter_prototype.__dict__['{k}'][0]"
        else:
            s_str = "{k}=parameter_prototype.__dict__['{k}']"
        arg_spec = ', '.join([s_str.format(k=k) for k in parameter_prototype.__dict__.keys() if not is_hidden(k)])
        if need_self:
            arg_spec = "self, " + arg_spec
        expr = \
            "def meta_fn({argspec}):\n" \
            "    pass\n" \
            "__signature__ = inspect.signature(meta_fn)\n".format(argspec=arg_spec)
        exec(expr, dict(inspect=inspect, Proto=parameter_prototype), locals())
        f.__signature__ = locals()['__signature__']
        return f

    return decorate
