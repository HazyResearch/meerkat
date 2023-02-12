from meerkat.interactive.graph import _reactive


@_reactive
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


@_reactive
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


@_reactive
def to_bool(x):
    """Overloaded ``bool`` operator.

    Use this when you want to use the ``bool`` operator on reactive values (e.g. Store).

    Args:
        x: The argument to convert to a bool.

    Returns:
        Store[bool] | bool: The result of the bool operation.
    """
    return bool(x)


@_reactive
def cnot(x):
    """Overloaded ``not`` operator.

    Use this when you want to use the ``not`` operator on reactive values (e.g. Store).

    Args:
        x: The arguments to not.

    Returns:
        The result of the and operation.
    """
    return not x
