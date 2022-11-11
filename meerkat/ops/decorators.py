from functools import wraps
from meerkat import DataFrame


def check_primary_key(fn: callable):
    """This decorator should wrap meerkat ops that could potentially invalidate a
    primary key. If the primary key is invalidated, the primary key is removed from
    the DataFrame.
    """
    @wraps(fn)
    def _wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)

        if isinstance(out, DataFrame) and out.primary_key is not None:
            if out._primary_key not in out or not out.primary_key._is_valid_primary_key():
                out.set_primary_key(None)
        return out 

    return _wrapper