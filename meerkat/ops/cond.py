from typing import Any

from meerkat.interactive.graph import react


@react()
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


@react()
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


@react()
def cnot(x):
    """Overloaded ``not`` operator.

    Use this when you want to use the ``not`` operator on reactive values (e.g. Store).

    Args:
        x: The arguments to not.

    Returns:
        The result of the and operation.
    """
    return not x


@react()
def _all(__iterable):
    return all(__iterable)


@react()
def _any(__iterable):
    return any(__iterable)


@react()
def _bool(x):
    """Overloaded ``bool`` operator.

    Use this when you want to use the ``bool`` operator on reactive values (e.g. Store).

    Args:
        x: The argument to convert to a bool.

    Returns:
        Store[bool] | bool: The result of the bool operation.
    """
    return bool(x)


@react()
def _complex(real: Any, imag: Any = 0.0) -> complex:
    if isinstance(real, str):
        return complex(real)
    return complex(real, imag)


@react()
def _int(__x, base: int = None):
    if base is None:
        return int(__x)
    return int(__x, base=base)


@react()
def _float(__x: Any) -> float:
    return float(__x)


@react()
def _len(__obj):
    return len(__obj)


@react()
def _hex(__number: Any) -> str:
    return hex(__number)


@react()
def _oct(__number: Any) -> str:
    return oct(__number)
