import os
from collections import ChainMap, defaultdict
from contextlib import suppress
from copy import copy
from inspect import cleandoc, isfunction, ismethod
from itertools import chain
from types import BuiltinFunctionType, SimpleNamespace
from warnings import warn

from expandvars import expandvars
from waterbear import Bear

from params_proto.parse_env_template import all_available
from params_proto.v2.utils import dot_to_deps


class Proto(SimpleNamespace):
  __value = None
  default = None
  help = None
  dtype = None

  def __init__(
    self,
    default=None,
    help=None,
    dtype=None,
    metavar="\b",
    env=None,
    strict_parsing=False,
    **kwargs,
  ):
    """
    The proto object. The env attribute allows one to set the environment variable
    from which this proto reads value from.

    :param default:
    :param help:
    :param dtype:
    :param metavar:
    :param env: the environment variable for the default value -- in the next version could be set
         automatically from the prefix of the class.
    :param strict_parsing: default to False, when true raises error for env var that are not set.
          this is passed onto the expandvars function as nounset.
    :param kwargs:
    """
    from termcolor import colored

    if default and not dtype:
      dtype = type(default)
    # only apply dtype to ENV, and when dtype is not None.
    if env:
      if strict_parsing or env in os.environ:
        default = os.environ[env]
      elif "$" in env and all_available(env, strict=True):
        # fall back to default, otherwise value becomes `''`.
        default = expandvars(env, nounset=strict_parsing) or default
      if dtype:
        default = dtype(default)

    help = cleandoc(help or "")

    default_str = str([default])[1:-1]
    if len(default_str) > 45:
      default_str = default_str[:42] + "..."
    default_str = default_str.replace("%", "%%")
    help_str = colored(f":{'any' if dtype is None else dtype.__name__} ", "blue")
    if env and env in os.environ:
      help_str += colored("$" + env, "magenta") + "="
    if default_str:
      help_str += colored(default_str, "cyan") + " "
    if help:
      # todo: handle multi-line help strings. Parse and remove indent.
      if len(help_str + help) > 60:
        help_str += "\n" + help.replace("%", "%%")
      else:
        help_str += help.replace("%", "%%")

    super().__init__(
      default=default, help=help_str, dtype=dtype, metavar=metavar, **kwargs
    )

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
    super().__init__(
      default=default,
      nargs=0,
      help=help,
      dtype=dtype,
      to_value=to_value,
      **kwargs,
    )


class Accumulant(Proto):
  """used by neo_hyper to avoid reset."""

  accumulant = True

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)


class StrType(type):
  thunk = None
  help = None

  def __matmul__(self, fn_str):
    return Eval(fn_str)


class Eval(Proto, metaclass=StrType):
  thunk = None

  def __init__(self, default, help=None, **kwargs):
    super().__init__(default=eval(default), nargs=0, help=help, dtype=eval, **kwargs)

  def __call__(self, *args, **kwargs):
    return self.thunk(*args, **kwargs)


def is_private(k: str) -> bool:
  """Filter Private Attributes.

  Args:
      k: <str> the key to be tested.

  Returns True if key equals "_prefix", or starts with "__" or "_<ParentClass>_"
  """
  return (
    k in ["_prefix", "_tree"]
    or k.startswith("_ParamsProto_")
    or k.startswith("_PrefixProto_")
    or k.startswith("_Meta_")
    or k.startswith("_Bear_")
    or k.startswith("__")
  )


def get_children(__init__):
  """decorator to parse the dependencies into current scope,
  using the cls._prefix attribute of the namespace.

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
      if not k.startswith("__"):
        setattr(cls, k, v)

    # note: This allows as to initialize ParamsProto on the class itself.
    #  super(ParamsProto, cls).__init__(cls)
    # Disable CLI for the base ParamsProto and PrefixProto classes
    if name in ("ParamsProto", "PrefixProto"):
      cli = False

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
  def _add_hooks(cls, hook, get_hook=None):
    cls.__set_hooks = (*cls.__set_hooks, hook)
    cls.__get_hooks = (*cls.__get_hooks, get_hook)

  def _pop_hooks(cls):
    cls.__set_hooks = cls.__set_hooks[:-1]
    cls.__get_hooks = cls.__get_hooks[:-1]

  def __setattr__(cls, item, value):
    if item == "_tree":
      return
    if item.startswith("_Meta_"):
      return type.__setattr__(cls, item, value)
    try:
      set_hooks = type.__getattribute__(cls, "_Meta__set_hooks")
      if callable(set_hooks[-1]):
        return set_hooks[-1](cls, item, value)
    except:
      return type.__setattr__(cls, item, value)

  def __getattribute__(self, item):
    if item.startswith("_Meta_"):
      return type.__getattribute__(self, item)
    try:
      get_hooks = type.__getattribute__(self, "_Meta__get_hooks")
      value = get_hooks[-1](self, item)
      assert value is not None
    except:
      value = type.__getattribute__(self, item)

    # rewrite in multiple lines
    if isinstance(value, Proto):
      return value.value
    # elif isinstance(value, property):
    #     return value.__get__(self)
    return value

  def _update(cls, __d: dict = None, **kwargs):
    """
    In-place update for the namespace. Useful for single-ton pattern.

    @param __d: positional-only argument, as a dot-separated dictionary.
    @param **kwargs: non-dot-separated keys (regular attribute), making it
       easy to update the data directly.
    @return:
    """
    if __d:
      if not cls._prefix:
        current_scope = {k: v for k, v in __d.items() if "." not in k}
      else:
        prefix = cls._prefix + "."
        current_scope = {
          k[len(prefix) :]: v for k, v in __d.items() if k.startswith(prefix)
        }

      for k, v in current_scope.items():
        if "." in k:
          first, rest = k.split(".", 1)
          getattr(cls, first)._update(current_scope)
        else:
          setattr(cls, k, v)

    for k, v in kwargs.items():
      if "." in k:
        raise RuntimeError(f"{k} is not supported via **kwargs in updates.")
      setattr(cls, k, v)

  @property  # used in the init of children
  def __vars__(cls):
    """
    recurrently return dictionary, only when the child has the same type.
    Only returns dictionary of children (but not grand children) if
    the child type is not ParamsProto.

    Returns: Nested Dict.
    """
    # note: support just one parent for now.
    lineage = [*find_ancestors(cls), super()]
    __vars = ChainMap(
      *[c.__vars__ if hasattr(c, "__vars__") else c.__dict__ for c in lineage]
    )

    d = {}
    for key in __vars.keys():
      if is_private(key):
        continue

      child = getattr(cls, key)

      if ismethod(child):
        continue
      if isfunction(child):
        continue
      if isinstance(child, BuiltinFunctionType):
        continue
      elif isinstance(child, staticmethod):
        # do not add static methods to the dictionary.
        # d[key] = child.__get__(None, cls)
        pass
      elif isinstance(child, property):
        # note: this is different from the instance method.
        # note-2: this is redundant if __getattribute__ also evaluates the properties on the class
        # d[key] = child.__get__(cls)
        d[key] = child
      # always recursive
      elif isinstance(child, ParamsProto) or isinstance(child, Bear):
        d[key] = child.__dict__
      else:
        d[key] = child

    return d

  @property  # has to be class property on ParamsProto
  def __dict__(cls):
    """
    recurrently return dictionary, only when the child has the same type.
    Only returns dictionary of children (but not grand children) if
    the child type is not ParamsProto.

    Returns: Nested Dict.
    """
    # note: support just one parent for now.
    __vars = ChainMap(*[c.__dict__ for c in find_ancestors(cls)], super().__dict__)

    d = {}
    for key in __vars.keys():
      if is_private(key):
        continue

      child = getattr(cls, key)

      if ismethod(child):
        continue
      if isfunction(child):
        continue
      if isinstance(child, BuiltinFunctionType):
        continue
      elif isinstance(child, staticmethod):
        # do not add static methods to the dictionary.
        # d[key] = child.__get__(None, cls)
        pass
      elif isinstance(child, property):
        # note: this is different from the instance method.
        # note-2: this is redundant if __getattribute__ also evaluates the properties on the class
        d[key] = child.__get__(cls)
      # always recursive
      elif isinstance(child, ParamsProto) or isinstance(child, Bear):
        d[key] = child.__dict__
      else:
        d[key] = child

    return d

  @property
  def _tree(cls):
    d = copy(cls.__dict__)
    for k, v in d.items():
      if isinstance(v, Meta):
        d[k] = v._tree
    return d

  def _register_args(cls, prefix=None):
    prefix = "" if not prefix else f"{prefix}."
    if cls._prefix:
      prefix = prefix + cls._prefix + "."

    doc_str = cleandoc(cls.__doc__ or "")

    if prefix:
      ARGS.add_argument_group(prefix, doc_str)
    else:
      desc = ARGS.parser.description or ""
      ARGS.parser.description = desc + doc_str + "\n"

    for k, v in super().__dict__.items():
      if is_private(k):
        continue

      # keys = [f"--{prefix}{k}"]
      # if "_" in keys[-1]:
      #     keys.append(f"--{prefix}{k.replace('_', '-')}")
      keys = [f"--{prefix}{k.replace('_', '-')}"]

      # fixme: this is surely wrong
      if is_subclass(v):
        v._register_args(cls._prefix)
      elif isinstance(v, Proto):
        ARGS.add_argument(cls, k, *keys, _argument_group=prefix, **vars(v))
      else:
        try:
          if issubclass(v, ParamsProto):
            v._register_args(cls._prefix)
          else:
            v = Proto(v)
            ARGS.add_argument(cls, k, *keys, _argument_group=prefix, **vars(v))
        except:
          v = Proto(v)
          ARGS.add_argument(cls, k, *keys, _argument_group=prefix, **vars(v))

    ARGS.last_group = None


import argparse


class ArgFactory:
  """The reason we do not inherit from argparse, is because the
  argument group returns new instances, so unless we patch those
  classes as well, we will not be able to intercept these calls.
  (I tried that first.)

  For this reason we implement this as a funciton, with a stateful
  context."""

  groups = defaultdict(lambda: None)
  last_group = None

  def __init__(
    self,
  ):
    fmt_cls = lambda prog: argparse.RawTextHelpFormatter(
      prog, indent_increment=4, max_help_position=50
    )
    self.parser = argparse.ArgumentParser(formatter_class=fmt_cls)
    self.__args = {}

  clear = __init__

  def add_argument(
    self,
    proto,
    key,
    *name_or_flags,
    _argument_group=None,
    default=None,
    dtype=None,
    to_value=None,
    **kwargs,
  ):
    local_args = {}
    if _argument_group:
      parser = self.groups[_argument_group]
    else:
      parser = self.last_group or self.parser
    for arg_key in name_or_flags:
      if arg_key in self.__args:
        warn(
          f"{arg_key} has already been registered. "
          f"This could be okay if intentional. Previous "
          f"value is {self.__args[arg_key]}. New value is "
          f"{kwargs}."
        )
        local_args[arg_key] = kwargs

    class BoolAction(argparse.Action):
      """parses 'true' to True, 'false' to False etc."""

      def __call__(self, parser, namespace, values, option_string):
        if values in ["false", "False"]:
          values = False
        elif values in ["true", "True"]:
          values = True
        elif values in ["none", "null", "None"]:
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
      parser.add_argument(
        *name_or_flags,
        default=default,
        type=str,
        dest=key,
        action=BoolAction,
        **kwargs,
      )
    else:
      parser.add_argument(
        *name_or_flags,
        default=default,
        type=dtype,
        dest=key,
        action=ArgAction,
        **kwargs,
      )
    self.__args.update(local_args)

  def add_argument_group(self, name, description):
    self.last_group = self.parser.add_argument_group(name, description)
    self.groups[name] = self.last_group

  def parse_args(self, *args):
    args, unknown = self.parser.parse_known_args()


ARGS = ArgFactory()  # this is the global store


# # todo: add a base asbtraction
# class Mixin(ParamsProto, cli=False):


class ParamsProto(Bear, metaclass=Meta):
  _prefix = "ParamsProto"  # b/c cls._prefix only created in subclass.

  def __init_subclass__(cls, prefix=False, **kwargs):
    super().__init_subclass__()
    if prefix is True:
      cls._prefix = cls.__name__
    elif isinstance(prefix, str):
      cls._prefix = prefix
    else:
      cls._prefix = None

  def __post_init__(self, _deps=None):
    pass

  def __new__(cls, _deps=None, _prefix=None, **children):
    ins = super(ParamsProto, cls).__new__(cls)
    # Note: initialize Bear without passing the children,
    #  because children might contain nested configs.
    return ins

  @get_children
  def __init__(self, _deps=None, _prefix=None, **children):
    """default init function, called after __new__."""
    # Note: grab the keys from Meta class--this is very clever. - Ge
    # Note: in fact we might not need to Bear class anymore.
    # todo: really want to change this behavior -- make children override by default??
    __vars = self.__class__.__vars__
    for key, child in __vars.items():
      if is_private(key):
        continue

      # this means that the new parameters do not
      if key not in children:
        cfg = child
      else:
        cfg = children[key]

      if is_subclass(child):
        children[key] = child(_deps, **cfg)
      elif is_subclass(child, ancestors=(Bear,)):
        # constructor should iteratively create children.
        children[key] = child(**cfg)
      elif isinstance(child, property):
        # fixme: ordering is important, and this could fail. Should run at the end.
        children[key] = child.__get__(self)
      else:
        children[key] = cfg

    super().__init__(_prefix=_prefix, __recursive=False, **children)

    prestine = True
    with suppress(TypeError):
      self.__post_init__(_deps)
      prestine = False
    if prestine:
      self.__post_init__()

  def __getattribute__(self, item):
    # todo: Makes more sense to do at compile time.
    value = Bear.__getattribute__(self, item)
    if isinstance(value, Proto):
      return value.value
    return value

  @property  # has to be class property on ParamsProto
  def __dict__(self):
    """
    NOT Recursive: return dictionary, only when the child has the same type.

    Properties should remain dynamic

    Returns: Dict.
    """
    # note: support just one parent for now.
    __vars = vars(self.__class__)
    # overwrite with older value, to avoid mutability
    cached = super().__dict__

    d = {}
    for key, child in cached.items():
      if is_private(key):
        continue
      if isinstance(child, property):
        d[key] = __vars[key]
      else:
        d[key] = child

    return d

  @property
  def _tree(self):
    """
    recurrently return dictionary, only when the child has the same type.
    Only returns dictionary of children (but not grand children) if
    the child type is not ParamsProto.

    Properties should remain dynamic

    Returns: Nested Dict.
    """
    # note: support just one parent for now.
    d = copy(self.__dict__)
    for key, child in d.items():
      if isinstance(child, ParamsProto):
        d[key] = child._tree

    return d


def find_ancestors(cls):
  """Find all ancestors of a class.

  Args:
      cls:

  Returns:

  """
  ancestors = []
  for c in cls.__bases__:
    if is_base_class(c, extra=(object,)):
      continue
    ancestors.append(c)
    ancestors.extend(find_ancestors(c))

  return ancestors


class PrefixProto(ParamsProto):
  """A ParamsProto class with prefix set to True.

  Since we override the __init_subclass__ method, the returned classes instance is
  still a ParamsProto class. NOT a PrefixProto class.
  """

  _prefix = "PrefixProto"

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(prefix=True, **kwargs)


def is_subclass(cls, *, ancestors=(ParamsProto, PrefixProto), extra=tuple()):
  for tc in chain(ancestors, extra):
    try:
      if issubclass(cls, tc):
        return True
    except:
      pass

  return False


def is_base_class(cls, extra=tuple()):
  """Check if a class is a base class of ParamsProto.

  Args:
      cls:

  Returns:

  """
  return cls in [ParamsProto, PrefixProto, Bear, *extra]


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
      setattr(Config, k[len(Config._prefix) + 1 :], v)
