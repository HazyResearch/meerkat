import functools
import inspect
from typing import Callable, Optional

from meerkat.ops.watch import WatchLogger


def errand(fn: Optional[Callable] = None, *, logger: WatchLogger):
    """A decorator for errands."""

    if fn is None:
        return functools.partial(errand, logger=logger)

    # Assert that one of the arguments of `fn` is `engine`.
    # If not, raise an error.
    signature = inspect.signature(fn)
    if "engine" not in signature.parameters:
        raise ValueError(f"Errands must have an argument called `engine`.")

    # Log the errand.
    errand_id = logger.log_errand(
        code=inspect.getsource(fn),
        name=fn.__name__,
        module=fn.__module__,
    )

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Bind the arguments to the function.
        bound = signature.bind(*args, **kwargs)

        # Get the engine.
        engine = bound.arguments.get("engine", signature.parameters['engine'].default)

        # Collect all the inputs i.e. all arguments except `engine`.
        inputs = {
            name: value for name, value in bound.arguments.items() if name != "engine"
        }

        # Log the errand start.
        errand_run_id = logger.log_errand_start(
            errand_id=errand_id,
            inputs=inputs,
            engine=engine.name,
        )

        # Run the errand.
        engine.set_logger(logger=logger)
        engine.set_errand_run_id(errand_run_id=errand_run_id)
        output = fn(*args, **kwargs)
        engine.set_logger(logger=None)
        engine.set_errand_run_id(errand_run_id=None)

        # Log the errand end.
        logger.log_errand_end(
            errand_run_id=errand_run_id,
            outputs={"output": output} if not isinstance(output, dict) else output,
        )

        return output

    return wrapper
