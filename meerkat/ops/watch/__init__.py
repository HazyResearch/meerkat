from functools import wraps
import inspect
from typing import Callable
from meerkat.engines import TextCompletion


def pretender(fn: Callable = None, engine: TextCompletion = None):
    """A decorator to implement a function using a language model."""

    if fn is None:
        return partial(pretender, engine=engine)

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
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Bind the arguments to the signature.
        bound_args = sig.bind(*args, **kwargs)
        # Fill in the template.
        prompt = template.format(
            source=str(inspect.getsource(fn)).replace("@pretender", ""),
            params=bound_args.arguments,
        )

        return eval(engine.run(prompt))

    return wrapper


if __name__ == "__main__":
    # Example usage.
    @pretender
    def fruits(n: int = 3) -> list[str]:
        """A list of `n` fruits."""
