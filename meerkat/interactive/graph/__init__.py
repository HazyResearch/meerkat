from typing import List

from tqdm import tqdm

from meerkat.interactive.graph.operation import Operation
from meerkat.interactive.graph.reactivity import (
    get_reactive_kwargs,
    is_reactive,
    no_react,
    react,
    reactive,
)
from meerkat.interactive.graph.store import (
    Store,
    StoreFrontend,
    make_store,
    store_field,
)
from meerkat.interactive.modification import Modification
from meerkat.interactive.node import _topological_sort
from meerkat.state import state
from meerkat.errors import TriggerError

__all__ = [
    "react",
    "no_react",
    "reactive",
    "is_reactive",
    "get_reactive_kwargs",
    "Store",
    "StoreFrontend",
    "make_store",
    "store_field",
    "Operation",
    "trigger",
]



def trigger() -> List[Modification]:
    """Trigger the computation graph of an interface based on a list of
    modifications.

    To force trigger, add the modifications to the modification queue.

    Return:
        List[Modification]: The list of modifications that resulted from running the
            computation graph.
    """
    modifications = state.modification_queue.queue

    # build a graph rooted at the stores and refs in the modifications list
    root_nodes = [mod.node for mod in modifications if mod.node is not None]

    # Sort the nodes in topological order, and keep the Operation nodes
    order = [
        node.obj
        for node in _topological_sort(root_nodes)
        if isinstance(node.obj, Operation)
    ]

    new_modifications = []
    if len(order) > 0:
        print(f"triggered pipeline: {'->'.join([node.fn.__name__ for node in order])}")
        with tqdm(total=len(order)) as pbar:
            # Go through all the operations in order: run them and add
            # their modifications
            # to the new_modifications list
            for op in order:
                pbar.set_postfix_str(f"Running {op.fn.__name__}")

                try:
                    mods = op()
                except Exception as e:
                    # TODO (sabri): Change this to a custom error type
                    raise TriggerError("Exception in trigger. " + str(e))

                # TODO: check this
                # mods = [mod for mod in mods if not isinstance(mod, StoreModification)]
                new_modifications.extend(mods)
                pbar.update(1)
        print("done")

    # Clear out the modification queue
    state.modification_queue.clear()
    return modifications + new_modifications
