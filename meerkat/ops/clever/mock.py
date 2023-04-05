import inspect
from functools import partial
from typing import Callable

from meerkat.engines import TextCompletion


def mock(fn: Callable = None, engine: TextCompletion = None):
    """A decorator to implement a function using a language model."""

    if fn is None:
        return partial(mock, engine=engine)

    if engine is None:
        raise ValueError("You must provide an engine.")

    # Inspect the function signature.
    sig = inspect.signature(fn)

    # Create a prompt template.
    template = """
Given the function signature:
{source}
where the parameters passed in are:
{params}

Pretend you are this function.
Generate a return value for this function in the requested format. Only
generate the output, and DO NOT WRITE ANY OTHER TEXT.

Output:\
"""

    def wrapper(*args, **kwargs):
        # Bind the arguments to the signature.
        bound_args = sig.bind(*args, **kwargs)
        # Fill in the template.
        prompt = template.format(
            source=str(inspect.getsource(fn)).replace("@mock", ""),
            params=bound_args.arguments,
        )

        return eval(engine.run(prompt))

    return wrapper


if __name__ == "__main__":
    # Example usage.
    @mock
    def fruits(n: int = 3) -> list[str]:
        """A list of `n` fruits."""

    @mock(engine=engine_mock.dummy("['apple', 'orange', 'banana']"))
    def fruits(n: int = 3) -> list[str]:
        """A list of `n` fruits."""
