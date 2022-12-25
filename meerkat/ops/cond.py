from meerkat.interactive.graph import Store, reactive


@reactive
def cand(*args):
    """Overloaded ``and`` operator.

    Use this when you want to use the and operator on reactive values (e.g. Store)

    Args:
        *args: The arguments to and together.

    Returns:
        The result of the and operation.
    """
    inputs = [x.value if isinstance(x, Store) else x for x in args]
    x = inputs[0]
    for y in inputs[1:]:
        x = x and y
    return x


@reactive
def cor(*args):
    """Overloaded ``or`` operator.

    Use this when you want to use the ``or`` operator on reactive values (e.g. Store)

    Args:
        *args: The arguments to ``or`` together.

    Returns:
        The result of the ``or`` operation.
    """
    inputs = [x.value if isinstance(x, Store) else x for x in args]
    x = inputs[0]
    for y in inputs[1:]:
        x = x or y
    return x


@reactive
def cnot(x):
    """Overloaded ``not`` operator.

    Use this when you want to use the ``not`` operator on reactive values (e.g. Store).

    Args:
        x: The arguments to not.

    Returns:
        The result of the and operation.
    """
    if isinstance(x, Store):
        x = x.value
    return not x
