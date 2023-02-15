import logging
from typing import List

from meerkat.errors import TriggerError
from meerkat.interactive.graph.marking import (
    is_unmarked_context,
    is_unmarked_fn,
    mark,
    unmarked,
)
from meerkat.interactive.graph.operation import Operation
from meerkat.interactive.graph.reactivity import is_reactive_fn, reactive
from meerkat.interactive.graph.store import Store, StoreFrontend, make_store
from meerkat.interactive.modification import Modification
from meerkat.interactive.node import _topological_sort
from meerkat.state import state

__all__ = [
    "reactive",
    "unmarked",
    "mark",
    "is_unmarked_fn",
    "reactive",
    "is_unmarked_context",
    "is_reactive_fn",
    "reactive",
    "get_reactive_kwargs",
    "Store",
    "StoreFrontend",
    "make_store",
    "Operation",
    "trigger",
]


logger = logging.getLogger(__name__)


@unmarked()
def trigger() -> List[Modification]:
    """Trigger the computation graph of an interface based on a list of
    modifications.

    To force trigger, add the modifications to the modification queue.

    Return:
        List[Modification]: The list of modifications that resulted from running the
            computation graph.
    """
    # Get all the modifications collected so far
    # This clears the modification queue
    modifications = state.modification_queue.clear()

    # Stop tracking modifications since we're triggering with what we have
    state.modification_queue.unready()

    logger.debug(f"Triggering on {modifications}")

    # build a graph rooted at the stores and refs in the modifications list
    root_nodes = [mod.node for mod in modifications if mod.node is not None]

    # Sort the nodes in topological order, and keep the Operation nodes
    # TODO: dynamically traverse and sort the graph instead of pre-sorting.
    # We need to do this to skip operations where inputs are not changed.
    order = [
        node.obj
        for node in _topological_sort(root_nodes)
        if isinstance(node.obj, Operation)
    ]

    new_modifications = []
    if len(order) > 0:
        logger.debug(
            f"Triggered pipeline: {'->'.join([node.fn.__name__ for node in order])}"
        )

        # Add the number of operations to the progress queue
        # TODO: this should be an object that contains other information
        # for the start of the progress bar
        state.progress_queue.add([op.fn.__name__ for op in order])
        # Go through all the operations in order: run them and add
        # their modifications to the new_modifications list
        for i, op in enumerate(order):
            # Add the operation name to the progress queue
            # TODO: this should be an object that contains other information
            # for the progress bar
            state.progress_queue.add(
                {"op": op.fn.__name__, "progress": int(i / len(order) * 100)}
            )

            try:
                mods = op()
            except Exception as e:
                # TODO (sabri): Change this to a custom error type
                raise TriggerError("Exception in trigger. " + str(e)) from e

            # TODO: check this
            # mods = [mod for mod in mods if not isinstance(mod, StoreModification)]
            new_modifications.extend(mods)

        state.progress_queue.add({"op": "Done!", "progress": 100})
        logger.debug("Done running trigger pipeline.")

    return modifications + new_modifications
