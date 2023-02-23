from functools import wraps
from typing import Any, Iterable

from meerkat.interactive.graph import reactive


@reactive()
def cand(*args):
    """Overloaded ``and`` operator.

    Use this when you want to use the and operator on reactive values (e.g. Store)

    Args:
        *args: The arguments to and together.

    Returns:
        The result of the and operation.
    """
    x = args[0]
    for y in args[1:]:
        x = x and y
    return x


@reactive()
def cor(*args):
    """Overloaded ``or`` operator.

    Use this when you want to use the ``or`` operator on reactive values (e.g. Store)

    Args:
        *args: The arguments to ``or`` together.

    Returns:
        The result of the ``or`` operation.
    """
    x = args[0]
    for y in args[1:]:
        x = x or y
    return x


@reactive()
def cnot(x):
    """Overloaded ``not`` operator.

    Use this when you want to use the ``not`` operator on reactive values (e.g. Store).

    Args:
        x: The arguments to not.

    Returns:
        The result of the and operation.
    """
    return not x


@reactive()
@wraps(all)
def _all(__iterable: Iterable[object]) -> bool:
    return all(__iterable)


@reactive()
@wraps(any)
def _any(__iterable) -> bool:
    return any(__iterable)


@reactive()
def _bool(x) -> bool:
    """Overloaded ``bool`` operator.

    Use this when you want to use the ``bool`` operator on reactive values (e.g. Store).

    Args:
        x: The argument to convert to a bool.

    Returns:
        Store[bool] | bool: The result of the bool operation.
    """
    return bool(x)


@reactive()
def _complex(real: Any, imag: Any = 0.0) -> complex:
    if isinstance(real, str):
        return complex(real)
    return complex(real, imag)


@reactive()
def _int(__x, base: int = None):
    if base is None:
        return int(__x)
    return int(__x, base=base)


@reactive()
def _float(__x: Any) -> float:
    return float(__x)


@reactive()
@wraps(len)
def _len(__obj):
    return len(__obj)


@reactive()
@wraps(hex)
def _hex(__number: Any) -> str:
    return hex(__number)


@reactive()
@wraps(oct)
def _oct(__number: Any) -> str:
    return oct(__number)


@reactive()
def _str(__obj) -> str:
    return str(__obj)


@reactive(nested_return=False)
def _list(__iterable) -> list:
    return list(__iterable)


@reactive(nested_return=False)
def _tuple(__iterable) -> tuple:
    return tuple(__iterable)


@reactive()
@wraps(sum)
def _sum(__iterable) -> float:
    return sum(__iterable)


@reactive()
def _dict(**kwargs) -> dict:
    return dict(**kwargs)


@reactive(nested_return=False)
def _set(__iterable) -> set:
    return set(__iterable)


@reactive()
def _range(*args) -> range:
    return range(*args)


@reactive()
@wraps(abs)
def _abs(__x) -> float:
    return abs(__x)


@reactive()
def _max(__iterable, *, key=None) -> Any:
    """Overloaded ``max`` operator."""
    return max(__iterable, key=key)


@reactive()
def _min(__iterable, *, key=None) -> Any:
    """Overloaded ``min`` operator."""
    return min(__iterable, key=key)


@reactive()
def _slice(*args):
    """Overloaded ``slice`` class."""
    return slice(*args)
