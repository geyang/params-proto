from textwrap import dedent
from types import SimpleNamespace
from warnings import warn

from params_proto.utils import dot_to_deps
from waterbear import Bear


class Proto(SimpleNamespace):
    __value = None
    default = None
    help = None
    dtype = None

    def __init__(self, default, help=None, dtype=None, metavar='\b', **kwargs):
        dtype = dtype or type(default)
        default_str = str([default])[1:-1]
        if len(default_str) > 45:
            default_str = default_str[:42] + "..."
        default_str = default_str.replace('%', '%%')
        help = f"<{dtype.__name__}>" + (f" {default_str}" if default_str else "") + (f" {help}" if help else "")
        super().__init__(default=default, help=help, dtype=dtype, metavar=metavar, **kwargs)

    @property
    def value(self):
        return self.__value or self.default

    @value.setter
    def value(self, value):
        self.__value = value

    # @property
    # def __dict__(self):
    #     return {k: v for k, v in super().__dict__.items() if not is_private(k)}


class Flag(Proto):
    def __init__(self, help=None, to_value=True, default=None, dtype=None, **kwargs):
        help = f"-> {str([to_value])[1:-1]}" + (f" {help}" if help else "")
        dtype = dtype or type(to_value) or type(default)
        super().__init__(default=default, nargs=0, help=help, dtype=dtype, **kwargs)
        self.to_value = to_value


class Accumulant(Proto):
    """used by neo_hyper to avoid reset."""

    accumulant = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def is_private(k: str) -> bool:
    return k.startswith('_prefix') or \
           k.startswith("_ParamsProto_") or \
           k.startswith("_Meta_") or \
           k.startswith("__")


def get_children(__init__):
    """decorator to parse the dependencies into current scope,
    using the cls.__prefix attribute of the namespace.

    Allows one to use _prefix to override to original class._prefix
    """

    def deco(self, _deps=None, **overrides):
        _prefix = overrides.get("_prefix", self.__class__._prefix)
        _ = dot_to_deps(_deps or {}, _prefix) if _prefix else dot_to_deps(_deps or {})
        _.update(overrides)
        res = __init__(self, _deps, **_)
        return res

    return deco


class Meta(type):
    _prefix: str

    def __init__(cls, name, bases, namespace, cli=True, cli_parse=True, **kwargs):
        for k, v in namespace.items():
            if not k.startswith('__'):
                setattr(cls, k, v)

        # note: This allows as to initialize ParamsProto on the class itself.
        #  super(ParamsProto, cls).__init__(cls)
        if cli:
            cls._register_args()

            # note: after this, the Meta.__init__ constructor is called.
            #  for this reason, we parse the args inside the Meta.__init__
            #  constructor.
            if cli_parse:
                ARGS.parse_args()

    __set_hooks = tuple()
    __get_hooks = tuple()

    # Note: These are the new methods supporting custom setter and getter override.
    def _add_hook(cls, hook):
        cls.__set_hooks = (*cls.__set_hooks, hook)

    def _pop_hooks(cls):
        cls.__set_hooks = cls.__set_hooks[:-1]

    def __setattr__(cls, item, value):
        if item == "_Meta__set_hooks":
            return type.__setattr__(cls, item, value)
        elif cls.__set_hooks:
            return cls.__set_hooks[-1](cls, item, value)
        else:
            return type.__setattr__(cls, item, value)

    def __getattr__(cls, item):
        # effectively not used, because the cls.__get_hook is empty.
        if cls.__get_hooks:
            return cls.__get_hooks[-1](cls, item)
        else:
            return type.__getattribute__(cls, item)

    def __getattribute__(self, item):
        # todo: Makes more sense to do at compile time.
        value = type.__getattribute__(self, item)
        return value.value if isinstance(value, Proto) else value

    def _update(cls, __d: dict = None, **kwargs):
        """
        In-place update for the namespace. Useful for single-ton pattern.

        @param __d: positional-only argument, as a dot-separated dictionary.
        @param **kwargs: non-dot-separated keys (regular attribute), making it
           easy to update the data directly.
        @return:
        """
        # todo: support nested update.
        if __d:
            for k, v in __d.items():
                # when the prefix does not exist
                if not cls._prefix:
                    setattr(cls, k, v)
                elif k.startswith(cls._prefix + "."):
                    setattr(cls, k[len(cls._prefix) + 1:], v)

        for k, v in kwargs.items():
            setattr(cls, k, v)

    @property  # falls back to the
    def __vars__(cls):
        """this is the original vars, return a dictionary of
        children, without recursively converting descendents
        to a dictionary."""
        return {k: v for k, v in super().__dict__.items() if not is_private(k)}

    @property  # has to be class property on ParamsProto
    def __dict__(cls):
        """
            recurrently return dictionary, only when the child has the same type.
            Only returns dictionary of children (but not grand children) if
            the child type is not ParamsProto.

            Returns: Nested Dict.
            """
        _ = {}
        for k, v in super().__dict__.items():
            if is_private(k):
                continue
            if isinstance(v, ParamsProto):
                _[k] = vars(v)
            elif isinstance(v, Proto):
                _[k] = v.value
            else:
                try:
                    if issubclass(v, ParamsProto):
                        _[k] = vars(v)
                    else:
                        _[k] = v
                except:
                    _[k] = v
        return _

    def _register_args(cls, prefix=None):

        prefix = "" if not prefix else f"{prefix}."
        if cls._prefix:
            prefix = prefix + cls._prefix + "."

        doc_str = dedent(cls.__doc__ or "")

        if prefix:
            ARGS.add_argument_group(prefix, doc_str)
        else:
            desc = ARGS.parser.description or ""
            ARGS.parser.description = desc + doc_str + "\n"

        for k, v in super().__dict__.items():
            if is_private(k):
                continue

            keys = [f"--{prefix}{k}"]
            if "_" in keys[-1]:
                keys.append(f"--{prefix}{k.replace('_', '-')}")

            if isinstance(v, ParamsProto):
                v._register_args(cls._prefix)
            elif isinstance(v, Proto):
                ARGS.add_argument(cls, k, *keys, **vars(v))
            else:
                try:
                    if issubclass(v, ParamsProto):
                        v._register_args(cls._prefix)
                    else:
                        v = Proto(v)
                        ARGS.add_argument(cls, k, *keys, **vars(v))
                except:
                    v = Proto(v)
                    ARGS.add_argument(cls, k, *keys, **vars(v))

        ARGS.group = None


import argparse


class ArgFactory:
    """The reason we do not inherit from argparse, is because the
    argument group returns new instances, so unless we patch those
    classes as well, we will not be able to intercept these calls.
    (I tried that first.)

    For this reason we implement this as a funciton, with a stateful
    context."""
    group = None

    def __init__(self, ):
        fmt_cls = lambda prog: argparse.RawTextHelpFormatter(prog, indent_increment=4, max_help_position=50)
        self.parser = argparse.ArgumentParser(formatter_class=fmt_cls)
        self.__args = {}

    clear = __init__

    def add_argument(self, proto, key, *name_or_flags, default=None, dtype=None, to_value=None, **kwargs):
        local_args = {}
        parser = self.group or self.parser
        for arg_key in name_or_flags:
            if arg_key in self.__args:
                warn(f"{arg_key} has already been registered. "
                     f"This could be okay if intentional. Previous "
                     f"value is {self.__args[arg_key]}. New value is "
                     f"{kwargs}.")
                local_args[arg_key] = kwargs

        class BoolAction(argparse.Action):
            """parses 'true' to True, 'false' to False etc. """
            def __call__(self, parser, namespace, values, option_string):
                if values in ['false', 'False']:
                    values = False
                elif values in ['true', 'True']:
                    values = True
                elif values in ['none', 'null', 'None']:
                    values = None
                try:
                    getattr(proto, key).value = to_value or values
                except AttributeError:
                    setattr(proto, key, to_value or values)

        class ArgAction(argparse.Action):
            def __call__(self, parser, namespace, values, option_string):
                try:
                    getattr(proto, key).value = to_value or values
                except AttributeError:
                    setattr(proto, key, to_value or values)

        if dtype == bool:
            parser.add_argument(*name_or_flags, default=default, type=str, dest=key, action=BoolAction, **kwargs)
        else:
            parser.add_argument(*name_or_flags, default=default, type=dtype, dest=key, action=ArgAction, **kwargs)
        self.__args.update(local_args)

    def add_argument_group(self, name, description):
        self.group = self.parser.add_argument_group(name, description)

    def parse_args(self, *args):
        args, unknown = self.parser.parse_known_args()


ARGS = ArgFactory()  # this is the global store


class ParamsProto(Bear, metaclass=Meta, cli=False):
    _prefix = "ParamsProto"  # b/c cls._prefix only created in subclass.

    def __init_subclass__(cls, prefix=False, **kwargs):
        super().__init_subclass__()
        if prefix is True:
            cls._prefix = cls.__name__
        elif isinstance(prefix, str):
            cls._prefix = prefix
        else:
            cls._prefix = None

    def __new__(cls, _deps=None, _prefix=None, **children):
        ins = super(ParamsProto, cls).__new__(cls)
        # Note: initialize Bear without passing the children,
        #  because children might contain nested configs.
        super(ParamsProto, ins).__init__(**cls.__vars__)
        return ins

    @get_children
    def __init__(self, _deps=None, **children):
        """default init function, called after __new__."""
        # Note: grab the keys from Meta class--this is very clever. - Ge
        # Note: in fact we might not need to Bear class anymore.
        # todo: really want to change this behavior -- make children override by default??
        _ = self.__class__.__vars__
        _.update(children)
        super().__init__(**_)

    def __getattribute__(self, item):
        # todo: Makes more sense to do at compile time.
        value = Bear.__getattribute__(self, item)
        return value.value if isinstance(value, Proto) else value

    @property  # has to be class property on ParamsProto
    def __dict__(self):
        """
        recurrently return dictionary, only when the child has the same type.
        Only returns dictionary of children (but not grand children) if
        the child type is not ParamsProto.

        Returns: Nested Dict.
        """
        _ = {}
        for k, v in super().__dict__.items():
            if is_private(k):
                continue
            if isinstance(v, ParamsProto):
                _[k] = v.__dict__
            elif isinstance(v, Proto):
                _[k] = v.value
            else:
                try:
                    if issubclass(v, ParamsProto):
                        _[k] = vars(v)
                    else:
                        _[k] = v
                except:
                    _[k] = v
        return _


class PrefixProto(ParamsProto, cli=False):
    """A ParamsProto class with prefix set to True."""
    _prefix = "PrefixProto"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(prefix=True, **kwargs)


from typing import Union


def update(Config: Union[type, Meta, ParamsProto], override):
    """Update a ParamsProto namespace, or instance

      by the override dictionary. Note the dictionary
          is dot.separated

      dot-keys are not yet implemented.

      Args:
          Config:
          override:

      Returns:

      """
    for k, v in override.items():
        if k.startswith(Config._prefix + "."):
            setattr(Config, k[len(Config._prefix) + 1:], v)
