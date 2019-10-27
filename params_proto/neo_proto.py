from waterbear import Bear

from params_proto.utils import dot_to_deps

import re


def is_hidden(k: str) -> bool:
    """return True is method is hidden"""
    return bool(re.match("^_.*$", k))


def is_private(k: str) -> bool:
    return k.startswith("_ParamsProto_") or \
           k.startswith("_Meta_") or \
           k.startswith("__")


def get_children(__init__):
    """decorator to parse the dependencies into current scope,
    using the cls._prefix attribute of the namespace.

    Allows one to use _prefix to override to original class._prefix
    """

    def deco(self, _deps=None, _prefix=None, **overrides):
        _ = dot_to_deps(_deps or {}, _prefix or self.__class__._ParamsProto__prefix)
        _.update(overrides)
        return __init__(self, _deps, **_)

    return deco


class Meta(type):
    def __init__(cls, name, bases, namespace, **kwargs):
        cls.__d = {k: v for k, v in namespace.items() if not k.startswith("__")}

    @property # has to be class property on ParamsProto
    def __dict__(cls):
        """
            recurrently return dictionary, only when the child has the same type.
            Only returns dictionary of children (but not grand children) if
            the child type is not ParamsProto.

            Returns: Nested Dict.
            """
        return {
            k: v.__dict__ if isinstance(v, ParamsProto) else v
            for k, v in super().__dict__.items()
            if not is_private(k)
        }


class ParamsProto(Bear, metaclass=Meta):

    def __init_subclass__(cls, prefix=None):
        super().__init_subclass__()
        cls.__prefix = prefix or cls.__name__
        # This allows as to initialize ParamsProto on the class itself.
        # super(ParamsProto, cls).__init__(cls)

    def __new__(cls, _deps=None, _prefix=None, **children):
        ins = super(ParamsProto, cls).__new__(cls)
        # Note: initialize Bear without passing the children,
        #  because children might contain nested configs.
        super(ParamsProto, ins).__init__(**vars(cls))
        return ins

    @get_children
    def __init__(self, _deps=None, **children):
        """default init function, called after __new__."""
        # Note: grab the keys from Meta class--this is very clever. - Ge
        # Note: in fact we might not need to Bear class anymore.
        _ = vars(self.__class__)
        _.update(children)
        super().__init__(**_)

