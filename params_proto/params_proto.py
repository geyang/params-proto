import argparse
from distutils import util
import inspect
import re

from typing import TypeVar, Union, Callable

from waterbear import DefaultBear, Bear


def is_hidden(k: str) -> bool:
    """return True is method is hidden"""
    return bool(re.match("^_.*$", k))


def props_to_dict(obj):
    """
      takes class or instance object and returns the attributes that are not
      hidden.

      :param obj:
      :return:

      Usage Example

          Python 3.6.2 | packaged by conda-forge | (default, Jul 23 2017,
          23:01:38)
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

      Attributes normally assigned to self inside `__init__` function shows up
      anyways.
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

    def __call__(self, *args, **kwargs):
        print(*args, **kwargs)

    @property
    def __dict__(self):
        """
            recurrently return dictionary, only when the child has the same type.
            Only returns dictionary of children (but not grand children) if
            the child type is not ParamsProto.

            Returns: Nested Dict.
            """
        return {
            k: v.__dict__ if isinstance(v, ParamsProto) else v
            for k, v in super().__dict__.items()
            if not is_hidden(k)
        }

    def __setattr__(self, key, value):
        prefix, *rest = key.split(".")
        import sys
        sys.stdout.flush()
        if len(rest) > 0:
            getattr(self, prefix).__setattr__(".".join(rest), value)
        else:
            super().__setattr__(key, value)


T = TypeVar("T")


def Proto(default: T, help=None, dtype=None, aliases=None, **kwargs) -> T:
    return DefaultBear(
        None,
        default=default,
        help_str=help,
        dtype=dtype,
        aliases=aliases,
        kwargs=kwargs)


def BoolFlag(default: bool, help=None, aliases=None, **kwargs) -> T:
    """this one generates a boolean flag that requires no arguments.

      The value of the flag is the opposite of the default.
      if default is True, then --bool-flag (such as --no-flag) would return False.
      if default is False, then (such as --render) would return True.
      """
    return Proto(default, help=help, dtype="bool-flag", aliases=aliases, **kwargs)


_bool = lambda v: v if v is None else bool(util.strtobool(v))

PREFIXES = []
LAZY = None
parser = None


def prefix_proto(prefix_or_fn: Union[str, None, Callable] = None, parse=False, **ext):
    global LAZY
    old, LAZY = LAZY, not parse

    # classes are callables.
    if callable(prefix_or_fn):
        fn = prefix_or_fn

        # only extend.
        for k, v in ext.items():
            if not hasattr(fn, k):
                setattr(fn, k, v)

        assert callable(fn), "The first input should be a function"
        assert hasattr(fn, "__name__"), "The function should have a name attribute."
        assert fn.__name__ is not None, ("Lambda Functions do not have names. "
                                         "Please use @prefix instead.")

        PREFIXES.append(fn.__name__)
        proto = cli_parse(fn)
        PREFIXES.pop(-1)

        LAZY = old
        return proto

    else:
        prefix = prefix_or_fn

        def _thunk(fn):
            global LAZY  # delayed reset.
            assert callable(fn), "The first input should be a function"

            if prefix is None:
                _prefix = prefix or fn.__name__
                assert fn.__name__ is not None, ("Lambda Functions do not have names. "
                                                 "Please use @prefix instead.")
            else:
                _prefix = prefix

            PREFIXES.append(_prefix)

            # only extend.
            for k, v in ext.items():
                if not hasattr(fn, k):
                    setattr(fn, k, v)

            proto = cli_parse(fn)
            PREFIXES.pop(-1)

            LAZY = old
            return proto

        return _thunk


from functools import partial, partialmethod

prefix_parse = partial(prefix_proto, parse=True)


# noinspection PyTypeChecker
def cli_parse(proto: T) -> T:
    """parser command line options, and repackage into a typed object.

      :param proto: T
      """
    parser = argparse.ArgumentParser(description=proto.__doc__)

    for k, p in proto.__dict__.items():
        if is_hidden(k):
            continue
        k_normalized = k.replace("_", "-")
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

        if data_type in [bool, "bool"]:
            data_type = _bool

        # do not parse non-leaf node
        if isinstance(p, ParamsProto):
            continue

        # so that support tuple/generator override.
        k_prefixed = ".".join(list(PREFIXES) + [k_normalized])

        if data_type == "bool-flag":
            parser.add_argument("--{k}".format(k=k_prefixed), *aliases, default=default_value,
                                action="store_false" if default_value else "store_true", help=help_str, **kwargs)
        elif data_type is list:
            parser.add_argument("--{k}".format(k=k_prefixed), *aliases, default=default_value,
                                nargs="*", type=data_type, help=help_str, **kwargs)
        else:
            parser.add_argument("--{k}".format(k=k_prefixed), *aliases, default=default_value,
                                type=data_type, help=help_str, **kwargs)

    params = ParamsProto(
        proto, **{k: v for k, v in vars(proto).items() if not is_hidden(k)})

    params.__parser = parser

    if not LAZY:
        parse(params, *PREFIXES)

    return params


proto = partial(prefix_proto, prefix=None, parse=False)


def parse(params, *prefixes):
    parser = params.__parser
    args, unknown_args = parser.parse_known_args()

    prefix = ".".join(prefixes)

    for k, v in vars(args).items():
        if k.startswith(prefix):
            setattr(params, k[len(prefix):], v)

    try:
        from argcomplete import autocomplete
        autocomplete(parser)
    except ImportError as e:
        print("failed to import argcomplete:", e)


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


def proto_partial(proto: ParamsProto, method=False):
    """Overrides the function with values from the Proto Object."""

    # todo(Ge): add support for fn(a, *, b, c...) for better control of
    #  the default values. Only the keyword arguments gets the default.
    #  ~
    #  replace partial and partialmethod with wrapper that updates the
    #  values.

    def wrap(f):
        ps = inspect.signature(f).parameters
        overrides = {}

        has_keyword_only = False
        for v in ps.values():
            if v.kind is v.KEYWORD_ONLY and v.default is v.empty:
                has_keyword_only = True

        for k, v in ps.items():
            if not hasattr(proto, k):
                continue
            if v.default is not v.empty:
                continue
            if has_keyword_only and v.kind is v.POSITIONAL_OR_KEYWORD:
                continue

            _ = getattr(proto, k)
            overrides[k] = _.default if hasattr(_, 'default') else _

        return (partialmethod if method else partial)(f, **overrides)

    return wrap
