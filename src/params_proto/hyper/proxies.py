"""Proxy classes for Sweep integration with v3 proto decorators."""
from typing import Any, Dict

from ..proto import ProtoWrapper


class BaseProxy:
    """Base proxy providing common Sweep interface."""

    def __init__(self, proto_obj):
        """Wrap a proto object."""
        object.__setattr__(self, "_proto", proto_obj)

    @property
    def _prefix(self):
        """Return prefix (to be implemented by subclasses)."""
        raise NotImplementedError

    def _add_hooks(self, set_hook, get_hook=None):
        """Add hooks to underlying proto object."""
        proto = object.__getattribute__(self, "_proto")
        proto._sweep_hooks.append(set_hook)

    def _pop_hooks(self):
        """Remove last hook from underlying proto object."""
        proto = object.__getattribute__(self, "_proto")
        if proto._sweep_hooks:
            proto._sweep_hooks.pop()

    def __getattr__(self, name):
        """Delegate to wrapped proto."""
        proto = object.__getattribute__(self, "_proto")
        return getattr(proto, name)

    def __setattr__(self, name, value):
        """Delegate to wrapped proto."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            proto = object.__getattribute__(self, "_proto")
            setattr(proto, name, value)


class ClassProxy(BaseProxy):
    """Proxy for @proto decorated classes (non-prefix)."""

    @property
    def _prefix(self):
        """Non-prefix classes don't have a prefix."""
        return None

    def _update(self, __d: Dict[str, Any] = None, **kwargs):
        """Update overrides."""
        proto = object.__getattribute__(self, "_proto")

        if __d:
            for k, v in __d.items():
                if "." not in k and k in proto._annotations:
                    proto._overrides[k] = v

        for k, v in kwargs.items():
            if k in proto._annotations:
                proto._overrides[k] = v

    def __dir__(self):
        """Return parameter names."""
        proto = object.__getattribute__(self, "_proto")
        return list(proto._annotations.keys())


class FuncProxy(BaseProxy):
    """Proxy for @proto.cli decorated functions (non-prefix)."""

    @property
    def _prefix(self):
        """Non-prefix functions don't have a prefix."""
        return None

    def _update(self, __d: Dict[str, Any] = None, **kwargs):
        """Update overrides."""
        proto = object.__getattribute__(self, "_proto")

        if __d:
            for k, v in __d.items():
                if "." not in k and k in proto._params:
                    proto._overrides[k] = v

        for k, v in kwargs.items():
            if k in proto._params:
                proto._overrides[k] = v

    def __dir__(self):
        """Return parameter names."""
        proto = object.__getattribute__(self, "_proto")
        return list(proto._params.keys())


class PrefixProxy(BaseProxy):
    """Proxy for @proto.prefix decorated classes/functions."""

    @property
    def _prefix(self):
        """Return kebab-case prefix."""
        proto = object.__getattribute__(self, "_proto")
        if isinstance(proto, ProtoWrapper):
            return proto._prefix  # Delegate to ProtoWrapper's _prefix property
        return None

    def _update(self, __d: Dict[str, Any] = None, **kwargs):
        """Update overrides with prefix support."""
        proto = object.__getattribute__(self, "_proto")
        prefix = self._prefix

        if __d:
            if prefix:
                prefix_key = f"{prefix}."
                for k, v in __d.items():
                    if k.startswith(prefix_key):
                        param_name = k[len(prefix_key):]
                        if isinstance(proto, ProtoWrapper):
                            if param_name in proto._params:
                                proto._overrides[param_name] = v
                    elif "." not in k:
                        if isinstance(proto, ProtoWrapper):
                            if k in proto._params:
                                proto._overrides[k] = v

        for k, v in kwargs.items():
            if isinstance(proto, ProtoWrapper):
                if k in proto._params:
                    proto._overrides[k] = v

    def __dir__(self):
        """Return parameter names."""
        proto = object.__getattribute__(self, "_proto")
        if isinstance(proto, ProtoWrapper):
            return list(proto._params.keys())
        return []
