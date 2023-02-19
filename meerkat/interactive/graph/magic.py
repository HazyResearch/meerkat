from functools import wraps
from typing import Any, Callable, List, TypeVar, cast

from meerkat.interactive.graph.marking import unmarked
from meerkat.interactive.graph.reactivity import reactive

# Used for annotating decorator usage of 'react'.
# Adapted from PyTorch:
# https://mypy.readthedocs.io/en/latest/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
T = TypeVar("T")


_IS_MAGIC_CONTEXT: List[bool] = []
_MAGIC_FN = "magic"


def _magic(fn: Callable) -> Callable:
    """Internal decorator that is used to mark a function with a wand. Wand
    functions can be activated when the magic context is active (`with
    magic:`).

    When the magic context is active, the function will be wrapped in
    `reactive` and will be executed as a reactive function.

    When the magic context is not active, the function will be effectively run
    with unmarked inputs (i.e. `with unmarked():`). This means that the function
    will not be added to the graph. The return value will be marked and returned.
    This allows the return value to be used as a marked element in the future.

    This is only meant for internal use.

    with mk.unmarked():
        with mk.magic():
            s = s + 3   # s.__add__ is a wand
    """

    def __wand(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            """Wrapper function that wraps the function in `reactive` if the
            magic context is active, and just wraps with `mark` otherwise."""

            # Check if magic context is active.
            if is_magic_context():
                # If active, we wrap with `reactive` and return.
                return reactive(fn, nested_return=False)(*args, **kwargs)
            else:
                # Just wrap with `mark` and return.
                with unmarked():
                    out = reactive(fn, nested_return=False)(*args, **kwargs)
                    # out = mark(out)
                return out

        # setattr(wrapper, "__wrapper__", _MAGIC_FN)
        return wrapper

    return __wand(fn)


class magic:
    """A context manager and decorator that changes the behavior of Store
    objects inside it. All methods, properties and public attributes of Store
    objects will be wrapped in @reactive decorators.

    Examples:
    """

    def __init__(self, magic: bool = True) -> None:
        self._magic = magic

    def __call__(self, func):
        @wraps(func)
        def decorate_context(*args, **kwargs):
            with self.clone():
                return func(*args, **kwargs)

        setattr(decorate_context, "__wrapper__", _MAGIC_FN)
        return cast(F, decorate_context)

    def __enter__(self):
        _IS_MAGIC_CONTEXT.append(self._magic)
        return self

    def __exit__(self, type, value, traceback):
        _IS_MAGIC_CONTEXT.pop(-1)

    def clone(self):
        return self.__class__(self._magic)


def is_magic_context() -> bool:
    """Whether the code is in a magic context.

    Returns:
        bool: True if the code is in a magic context.
    """
    # By default, we should not assume we are in a magic context.
    # This will mean that Store objects will default to not decorating
    # their methods and properties with @reactive.
    if len(_IS_MAGIC_CONTEXT) == 0:
        return False

    # Otherwise, we check if the user has explicitly used the
    # `magic` context manager or decorator.
    return len(_IS_MAGIC_CONTEXT) > 0 and bool(_IS_MAGIC_CONTEXT[-1])
