import asyncio
import concurrent
import functools
import warnings
from typing import Any, Callable, Union

from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm

import meerkat as mk


async def callasync(fn: Callable, element: Any):
    """
    Call a function asynchronously.

    Uses the `asyncio` library to call a function asynchronously.
    """
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, fn, element)
    return result


def asasync(fn: Callable):
    """
    Decorator to make a function asynchronous.
    """

    @functools.wraps(fn)
    async def wrapper(element):
        return await callasync(fn, element)

    return wrapper


async def apply_function(fn: Callable, column: mk.Column):
    """
    Asynchronously apply a function to each element in a column.
    Run the return value of this function through `asyncio.run()` to
    get the results.

    Args:
        fn: The async function to apply to each element.
        column: The column to apply the function to.
    """
    assert asyncio.iscoroutinefunction(fn), "Function must be async."
    tasks = []
    # for element in tqdm_asyncio(column, desc=f"Async execution of `{fn.__name__}`"):
    for element in column:
        tasks.append(fn(element))
    results = await asyncio.gather(*tasks)
    return results


def as_single_arg_fn(fn: Callable) -> Callable:
    """
    Convert a function that takes multiple arguments to a function that takes
    a single argument.
    """

    @functools.wraps(fn)
    def wrapper(kwargs):
        return fn(**kwargs)

    return wrapper


async def amap(
    data: Union[mk.DataFrame, mk.Column],
    function: Callable,
    pbar: bool = True,
    max_concurrent: int = 100,
) -> Union[mk.DataFrame, mk.Column]:
    """Apply a function to each element in the column.

    Args:
        data: The column or dataframe to apply the function to.
        function: The function to apply to each element in the column.
        pbar: Whether to show a progress bar or not.
        max_concurrent: The maximum number of concurrent tasks to run.

    Returns:
        A column or dataframe with the function applied to each element.
    """
    # # Check if the function is async. If not, make it.
    # if not asyncio.iscoroutinefunction(function):
    #     warnings.warn(
    #         f"Function {function} is not async. Automatically converting to async."
    #         " Consider making it async before calling `amap`."
    #     )

    if isinstance(data, mk.Column):
        # Run the function asynchronously on the column.
        # Only run `max_concurrent` tasks at a time.
        # Split the column into chunks of size `max_concurrent`.
        chunks = [
            data[i : i + max_concurrent] for i in range(0, len(data), max_concurrent)
        ]
        # Synchonously apply the function to each chunk.
        results = []
        for chunk in tqdm(chunks, desc=f"Chunked execution of `{function.__name__}`"):
            results.append(await apply_function(asasync(function), chunk))
        # Flatten the results.
        results = [item for sublist in results for item in sublist]
        # Create a new column with the results.
        return mk.ScalarColumn(results)

    elif isinstance(data, mk.DataFrame):
        # Run the function asynchronously on the dataframe.
        # Only run `max_concurrent` tasks at a time.
        # Get the function parameter names.
        import inspect

        parameter_names = list(inspect.signature(function).parameters.keys())

        # Restrict the dataframe to only the columns that are needed.
        data = data[parameter_names]

        # Split the dataframe into chunks of size `max_concurrent`.
        chunks = [
            data[i : i + max_concurrent] for i in range(0, len(data), max_concurrent)
        ]
        # Synchonously apply the function to each chunk.
        results = []
        for chunk in tqdm(chunks, desc=f"Chunked execution of `{function.__name__}`"):
            results.append(
                await apply_function(asasync(as_single_arg_fn(function)), chunk)
            )
        # Flatten the results.
        results = [item for sublist in results for item in sublist]
        # Create a new column with the results.
        return mk.ScalarColumn(results)
    else:
        raise TypeError(f"Data must be a Column or DataFrame. Got {type(data)}.")
