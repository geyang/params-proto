from types import SimpleNamespace

from params_proto.utils import dot_to_deps
from waterbear import Bear


class Proto(SimpleNamespace):
    __value = None
    default = None
    help = None
    dtype = None
    aliases = None
    kwargs = dict()

    def __init__(self, default, help=None, dtype=None, aliases=None, kwargs=None):
        super().__init__(default=default, help=help, dtype=dtype or type(default), aliases=aliases, kwargs=kwargs or {})

    @property
    def value(self):
        return self.__value or self.default

    @value.setter
    def value(self, value):
        self.__value = value


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
        _prefix = overrides.get("_prefix", None)
        _ = dot_to_deps(_deps or {}, _prefix or self.__class__._prefix)
        _.update(overrides)
        res = __init__(self, _deps, **_)
        return res

    return deco


class Meta(type):
    _prefix: str
    __set_hook = tuple()
    __get_hook = tuple()

    # Note: These are the new methods supporting custom setter and getter override.
    def _add_hook(cls, hook):
        cls.__set_hook = (*cls.__set_hook, hook)

    def _pop_hooks(cls):
        cls.__set_hook = cls.__set_hook[:-1]

    def __setattr__(cls, item, value):
        if item == "_Meta__set_hook":
            return type.__setattr__(cls, item, value)
        elif cls.__set_hook:
            return cls.__set_hook[-1](cls, item, value)
        else:
            return type.__setattr__(cls, item, value)

    def __getattr__(cls, item):
        # effectively not used, because the cls.__get_hook is empty.
        if cls.__get_hook:
            return cls.__get_hook[-1](cls, item)
        else:
            return type.__getattribute__(cls, item)

    def __getattribute__(self, item):
        # todo: Makes more sense to do at compile time.
        value = type.__getattribute__(self, item)
        return value.value if isinstance(value, Proto) else value

    def __init__(cls, name, bases, namespace, **kwargs):
        # cls.__namespace = {k: v for k, v in namespace.items() if not k.startswith("__")}
        for k, v in namespace.items():
            if not k.startswith('__'):
                setattr(cls, k, v)

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
                if k.startswith(cls._prefix + "."):
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


class ParamsProto(Bear, metaclass=Meta):
    _prefix = "ParamsProto"  # b/c cls._prefix only created in subclass.

    def __init_subclass__(cls, prefix=None):
        super().__init_subclass__()
        cls._prefix = cls.__name__ if prefix is None else prefix
        # This allows as to initialize ParamsProto on the class itself.
        # super(ParamsProto, cls).__init__(cls)

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
