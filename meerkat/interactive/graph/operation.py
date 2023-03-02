import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

from meerkat.interactive.graph.marking import unmarked
from meerkat.interactive.graph.utils import _replace_nodes_with_nodeables
from meerkat.interactive.modification import (
    DataFrameModification,
    Modification,
    StoreModification,
)
from meerkat.interactive.node import NodeMixin
from meerkat.interactive.types import Primitive

if TYPE_CHECKING:
    from meerkat.interactive.graph.store import Store

logger = logging.getLogger(__name__)


def _check_fn_has_leading_self_arg(fn: Callable):
    """# FIXME: super hacky

    # We need to figure out why Store.__eq__ (and potentially other
    dunder methods) # are passed into `reactive` as the class method
    instead of the instance method. # In the meantime, we can check if
    the first argument is `self` and if so, # we can assume that the
    function is an instance method.
    """
    import inspect

    parameters = list(inspect.signature(fn).parameters)
    if len(parameters) > 0:
        return "self" == parameters[0]
    return False


class Operation(NodeMixin):
    def __init__(
        self,
        fn: Callable,
        args: List[Any],
        kwargs: Dict[str, Any],
        result: Any,
        skip_fn: Callable[..., bool] = None,
    ):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.result = result

        self.skip_fn = skip_fn
        self.cache = None
        if self.skip_fn is None:
            self._skip_parameters = None
        else:
            # The fn parameters we need to extract for skip_fn.
            self._skip_parameters = _validate_and_extract_skip_fn(
                skip_fn=self.skip_fn, fn=self.fn
            )
            self._reset_cache()

    def __repr__(self):
        return (
            f"Operation({self.fn.__name__}, {self.args}, {self.kwargs}, {self.result})"
        )

    def _reset_cache(self):
        """Set the cache to the new arguments and keyword arguments."""
        args, kwargs = self._dereference_nodes()
        cache = inspect.getcallargs(self.fn, *args, **kwargs)
        self.cache = {param: cache[param] for param in self._skip_parameters}

    def _run_skip_fn(self, *args, **kwargs):
        """Run the skip_fn to determine if the operation should be skipped.

        Args:
            *args: The new arguments to the function.
            **kwargs: The new keyword arguments to the function.
        """
        # Process cache.
        skip_kwargs = {f"old_{k}": v for k, v in self.cache.items()}
        self.cache = {}

        # Process new arguments.
        new_kwargs = inspect.getcallargs(self.fn, *args, **kwargs)
        new_kwargs = {
            f"new_{param}": new_kwargs[param] for param in self._skip_parameters
        }
        skip_kwargs.update(new_kwargs)

        skip = self.skip_fn(**skip_kwargs)
        logger.debug(f"Operation({self.fn.__name__}): skip_fn -> {skip}")
        return skip

    def _dereference_nodes(self):
        # Dereference the nodes.
        args = _replace_nodes_with_nodeables(self.args, unwrap_stores=True)
        kwargs = _replace_nodes_with_nodeables(self.kwargs, unwrap_stores=True)

        # Special logic to make sure we unwrap all Store objects, except those
        # that correspond to `self`.
        if _check_fn_has_leading_self_arg(self.fn):
            args = list(args)
            args[0] = _replace_nodes_with_nodeables(self.args[0], unwrap_stores=False)
        return args, kwargs

    def __call__(self) -> List[Modification]:
        """Execute the operation. Unpack the arguments and keyword arguments
        and call the function. Then, update the result Reference with the
        result and return a list of modifications.

        These modifications describe the delta changes made to the
        result Reference, and are used to update the state of the GUI.
        """

        logger.debug(f"Running {repr(self)}")

        # Dereference the nodes.
        args, kwargs = self._dereference_nodes()

        # If we are skipping the function, then we can
        # return an empty list of modifications.
        if self.skip_fn is not None:
            skip = self._run_skip_fn(*args, **kwargs)
            # Reset the cache to the new arguments and keyword arguments.
            # FIXME: We are deferencing the nodes again, which we dont need to do.
            self._reset_cache()
            if skip:
                return []

        with unmarked():
            update = self.fn(*args, **kwargs)

        modifications = []
        self.result = _update_result(self.result, update, modifications=modifications)

        return modifications


def _update_result(
    result: Union[list, tuple, dict, "Store", Primitive],
    update: Union[list, tuple, dict, "Store", Primitive],
    modifications: List[Modification],
) -> Union[list, tuple, dict, "Store", Primitive]:
    """Update the result object with the update object. This recursive function
    will perform a nested update to the result with the update. This function
    will also update the modifications list with the changes made to the result
    object.

    Args:
        result: The result object to update.
        update: The update object to use to update the result.
        modifications: The list of modifications to update.

    Returns:
        The updated result object.
    """
    from meerkat.dataframe import DataFrame
    from meerkat.interactive.graph.store import Store

    if isinstance(result, DataFrame):
        # Detach the result object from the Node
        inode = result.detach_inode()

        # Attach the inode to the update object
        update.attach_to_inode(inode)

        # Create modifications
        modifications.append(DataFrameModification(id=inode.id, scope=update.columns))

        return update

    elif isinstance(result, Store):
        # If the result is a Store, then we need to update the Store's value
        # and return a StoreModification
        # TODO(karan): now checking if the value is the same
        # This is assuming that all values put into Stores have an __eq__ method
        # defined that can be used to check if the value has changed.
        if isinstance(result, (str, int, float, bool, type(None), tuple)):
            # We can just check if the value is the same
            if result.value != update:
                result.set(update)
                modifications.append(
                    StoreModification(id=result.inode.id, value=update)
                )
        else:
            # We can't just check if the value is the same if the Store contains
            # a list, dict or object, since they are mutable (and it would just
            # return True).
            result.set(update)
            modifications.append(StoreModification(id=result.inode.id, value=update))
        return result
    elif isinstance(result, list):
        # Recursively update each element of the list
        return [_update_result(r, u, modifications) for r, u in zip(result, update)]
    elif isinstance(result, tuple):
        # Recursively update each element of the tuple
        return tuple(
            _update_result(r, u, modifications) for r, u in zip(result, update)
        )
    elif isinstance(result, dict):
        # Recursively update each element of the dict
        return {
            k: _update_result(v, update[k], modifications) for k, v in result.items()
        }
    else:
        # If the result is not a Reference or Store, then it is a primitive type
        # and we can just return the update
        return update


def _validate_and_extract_skip_fn(*, skip_fn, fn) -> set:
    # Skip functions should have arguments that start with `old_` and `new_`
    # followed by the same keyword argument name.
    # e.g. def skip(old_x, new_x):
    fn_parameters = inspect.signature(fn).parameters.keys()
    skip_parameters = inspect.signature(skip_fn).parameters.keys()

    if not all(
        param.startswith("old_") or param.startswith("new_")
        for param in skip_parameters
    ):
        raise ValueError(
            f"Expected skip_fn to have parameters that start with "
            f"`old_` or `new_`, but got parameters: {skip_parameters}"
        )

    # Remove the `old_` and `new_` prefixes from the skip parameters.
    skip_parameters_no_prefix = {param[4:] for param in skip_parameters}
    if not skip_parameters_no_prefix.issubset(fn_parameters):
        unknown_parameters = skip_parameters_no_prefix - fn_parameters
        unknown_parameters = [x for x in skip_parameters if x[4:] in unknown_parameters]
        unknown_parameters = ", ".join(unknown_parameters)
        raise ValueError(
            "Expected skip_fn to have parameters that match the "
            "parameters of the function, but got parameters: "
            f"{skip_parameters_no_prefix}"
        )

    return skip_parameters_no_prefix
