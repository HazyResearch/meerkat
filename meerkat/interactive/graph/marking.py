from functools import wraps
from typing import Any, Callable, List, TypeVar, cast

# Used for annotating decorator usage of 'react'.
# Adapted from PyTorch:
# https://mypy.readthedocs.io/en/latest/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
T = TypeVar("T")

_IS_UNMARKED_CONTEXT: List[bool] = []
_UNMARKED_FN = "unmarked"


class unmarked:
    """A context manager and decorator that forces all objects within it to
    behave as if they are not marked. This means that any functions (reactive
    or not) called with those objects will never be rerun.

    Effectively, functions (by decoration) or blocks of code
    (with the context manager) behave as if they are not reactive.

    Examples:

    Consider the following function:

        >>> @reactive
        ... def f(x):
        ...     return x + 1

    If we call `f` with a marked object, then it will be rerun if the
    object changes:

        >>> x = mark(1)
        >>> f(x) # f is rerun when x changes

    Now, suppose we call `f` inside another function `g` that is not
    reactive:

        >>> def g(x):
        ...     out = f(x)
        ...     return out

    If we call `g` with a marked object, then the `out` variable will be
    recomputed if the object changes. Even though `g` is not reactive,
    `f` is, and `f` is called within `g` with a marked object.

    Sometimes, this might be what we want. However, sometimes we want
    to ensure that a function or block of code behaves as if it is not
    reactive.

    For this behavior, we can use the `unmarked` context manager:

        >>> with unmarked():
        ...     g(x) # g and nothing in g is rerun when x changes

    Or, we can use the `unmarked` decorator:

        >>> @unmarked
        ... def g(x):
        ...     out = f(x)
        ...     return out

    In both cases, the `out` variable will not be recomputed if the object
    changes, even though `f` is reactive.
    """

    def __call__(self, func):
        from meerkat.interactive.graph.reactivity import reactive

        @wraps(func)
        def decorate_context(*args, **kwargs):
            with self.clone():
                return reactive(func, nested_return=False)(*args, **kwargs)

        setattr(decorate_context, "__wrapper__", _UNMARKED_FN)
        return cast(F, decorate_context)

    def __enter__(self):
        _IS_UNMARKED_CONTEXT.append(True)
        return self

    def __exit__(self, type, value, traceback):
        _IS_UNMARKED_CONTEXT.pop(-1)

    def clone(self):
        return self.__class__()


def is_unmarked_context() -> bool:
    """Whether the code is in an unmarked context.

    Returns:
        bool: True if the code is in an unmarked context.
    """
    # By default, we should not assume we are in an unmarked context.
    # This will allow functions that are decorated with `reactive` to
    # add nodes to the graph.
    if len(_IS_UNMARKED_CONTEXT) == 0:
        return False

    # TODO: we need to check this since users are only allowed the use
    # of the `unmarked` context manager. Therefore, everything is reactive
    # by default, *unless the user has explicitly used `unmarked`*.
    return len(_IS_UNMARKED_CONTEXT) > 0 and bool(_IS_UNMARKED_CONTEXT[-1])


def is_unmarked_fn(fn: Callable) -> bool:
    """Check if a function is wrapped by the `unmarked` decorator."""
    return (
        hasattr(fn, "__wrapped__")
        and hasattr(fn, "__wrapper__")
        and fn.__wrapper__ == _UNMARKED_FN
    )


def mark(input: T) -> T:
    """Mark an object.

    If the input is an object, then the object will become reactive: all of its
    methods and properties will become reactive. It will be returned as a
    `Store` object.

    Args:
        input: Any object to mark.

    Returns:
        A reactive function or object.

    Examples:

    Use `mark` on primitive types:

        >>> x = mark(1)
        >>> # x is now a `Store` object

    Use `mark` on complex types:

        >>> x = mark([1, 2, 3])

    Use `mark` on instances of classes:

        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2, 3]})
        >>> x: Store = mark(df)
        >>> y = x.head()

        >>> class Foo:
        ...     def __init__(self, x):
        ...         self.x = x
        ...     def __call__(self):
        ...         return self.x + 1
        >>> f = Foo(1)
        >>> x = mark(f)

    Use `mark` on functions:

        >>> aggregation = mark(mean)
    """
    from meerkat.interactive.graph.store import Store
    from meerkat.mixins.reactifiable import MarkableMixin

    if isinstance(input, MarkableMixin):
        return input.mark()
    return Store(input)
