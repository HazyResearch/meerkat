from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

from meerkat.dataframe import DataFrame
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


class Operation(NodeMixin):
    def __init__(
        self,
        fn: Callable,
        args: List[Any],
        kwargs: Dict[str, Any],
        result: Any,
    ):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.result = result

    def __repr__(self):
        return (
            f"Operation({self.fn.__name__}, {self.args}, {self.kwargs}, {self.result})"
        )

    def __call__(self) -> List[Modification]:
        """Execute the operation. Unpack the arguments and keyword arguments
        and call the function. Then, update the result Reference with the
        result and return a list of modifications.

        These modifications describe the delta changes made to the
        result Reference, and are used to update the state of the GUI.
        """
        # Dereference the nodes.
        args = _replace_nodes_with_nodeables(self.args)
        kwargs = _replace_nodes_with_nodeables(self.kwargs)

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
            if result != update:
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
