import argparse
from argcomplete import autocomplete
from distutils import util
import inspect
import re

from typing import TypeVar

from waterbear import DefaultBear, Bear


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


class ParamsProto(DefaultBear):
    """Parameter Prototype Class, has toDict method and __proto__ attribute for the original namespace object."""

    def __init__(self, proto, **d):
        super().__init__(None, **d)
        self._proto = proto

    @property
    def __dict__(self):
        return {k: v for k, v in super().__dict__.items() if not is_hidden(k)}


T = TypeVar('T')


def Proto(default: T, help=None, dtype=None, aliases=None, **kwargs) -> T:
    return DefaultBear(None, default=default, help_str=help, dtype=dtype, aliases=aliases, kwargs=kwargs)


def BoolFlag(default: bool, help=None, aliases=None, **kwargs) -> T:
    """this one generates a boolean flag that requires no arguments.
    The value of the flag is the opposite of the default.
    if default is True, then --bool-flag (such as --no-flag) would return False.
    if default is False, then (such as --render) would return True.
    """
    return Proto(default, help=help, dtype="bool-flag", aliases=aliases, **kwargs)


_bool = lambda v: v if v is None else bool(util.strtobool(v))


# noinspection PyTypeChecker
def cli_parse(proto: T) -> T:
    """parser command line options, and repackage into a typed object.
    :type proto: T
    """
    parser = argparse.ArgumentParser(description=proto.__doc__)
    for k, p in proto.__dict__.items():
        if is_hidden(k):
            continue
        k_normalized = k.replace('_', '-')
        if type(p) is DefaultBear:
            default_value = p.default
            help_str = p.help_str or "N/A"
            data_type = p.dtype or type(default_value)
            kwargs = p.kwargs
            aliases = p.aliases or []
        else:
            default_value = p
            help_str = "N/A"
            data_type = type(p)
            kwargs = {}
            aliases = []

        if data_type in [bool, 'bool']:
            data_type = _bool

        if data_type == "bool-flag":
            parser.add_argument('--{k}'.format(k=k_normalized), *aliases, default=default_value,
                                action="store_false" if default_value else "store_true",
                                help=help_str, **kwargs)
        elif data_type is list:
            parser.add_argument('--{k}'.format(k=k_normalized), *aliases, default=default_value, nargs="*",
                                type=data_type, help=help_str, **kwargs)
        else:
            parser.add_argument('--{k}'.format(k=k_normalized), *aliases, default=default_value,
                                type=data_type, help=help_str, **kwargs)

    params = ParamsProto(proto, **{k: v for k, v in vars(proto).items() if not is_hidden(k)})

    args, unknow_args = parser.parse_known_args()
    # logging.debug("params_proto: args: {}\n              unknow_args: {}", args, unknow_args)
    params.update(vars(args))

    autocomplete(parser)

    return params


def get_default(p):
    if type(p) is DefaultBear:
        return p.default
    else:
        return p


# noinspection PyUnusedLocal
def proto_signature(parameter_prototype, need_self=False):
    def decorate(f):
        # Need to have return type as well.
        __doc__ = f.__doc__

        s_str = "{k}={default}"
        arg_spec = ', '.join([s_str.format(k=k, default=get_default(p))
                              for k, p in parameter_prototype.__dict__.items() if not is_hidden(k)])
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
