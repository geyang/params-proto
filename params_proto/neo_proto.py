from waterbear import Bear

from params_proto.utils import dot_to_deps


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
    def __init__(cls, name, bases, namespace, **kwargs):
        # cls.__namespace = {k: v for k, v in namespace.items() if not k.startswith("__")}
        for k, v in namespace.items():
            if not k.startswith('__'):
                setattr(cls, k, v)

    @property  # falls back to the
    def __vars__(cls):
        """this is the original vars, return a dictionary of
        children, without recursively converting descendents
        to a dictionary."""
        return {k: v for k, v in super().__dict__.items() if not is_private(k)}

    # Note: type and object do not have __getattr__. They only
    #  have __getattribute__.
    # def __getattribute__(self, item):
    #     try:
    #         return type.__getattribute__(self, item)
    #     except AttributeError:
    #         return self.__namespace['_prefix']
    #     # if item == "_prefix":
    #     #     return self.__namespace['_prefix']
    #     # else:
    #     #     return type.__getattribute__(self, item)

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
                _[k] = v.__dict__
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
        _ = self.__class__.__vars__
        _.update(children)
        super().__init__(**_)

    def __getattr__(self, item):
        if item == "_prefix":
            return self._prefix
        else:
            return super().__getattr__(item)

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
            else:
                try:
                    if issubclass(v, ParamsProto):
                        _[k] = vars(v)
                    else:
                        _[k] = v
                except:
                    _[k] = v
        return _
