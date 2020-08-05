import inspect
from functools import partialmethod, wraps

from .neo_proto import ParamsProto


def proto_partial(proto: ParamsProto, method=False):
    """Overrides the function with values from the Proto Object.
    """

    def decorate(f):
        ps = inspect.signature(f).parameters

        has_keyword_only = False
        for v in ps.values():
            if v.kind is v.KEYWORD_ONLY and v.default is v.empty:
                has_keyword_only = True

        @wraps(f)
        def partial(*args, **kwargs):
            overrides = {}
            for k, v in ps.items():
                if not hasattr(proto, k):
                    continue
                if v.default is not v.empty:
                    continue
                if has_keyword_only and v.kind is v.POSITIONAL_OR_KEYWORD:
                    continue

                _ = getattr(proto, k)
                overrides[k] = _.default if hasattr(_, 'default') else _

            overrides.update(kwargs)
            return f(*args, **overrides)

        if method:
            return partialmethod(partial)
        return partial

    return decorate
