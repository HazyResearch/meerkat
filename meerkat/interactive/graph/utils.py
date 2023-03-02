from typing import List

from meerkat.interactive.node import Node, NodeMixin


def _replace_nodeables_with_nodes(obj):
    if isinstance(obj, NodeMixin):
        obj = obj.inode
    elif isinstance(obj, list) or isinstance(obj, tuple):
        obj = type(obj)(_replace_nodeables_with_nodes(x) for x in obj)
    elif isinstance(obj, dict):
        obj = {
            _replace_nodeables_with_nodes(k): _replace_nodeables_with_nodes(v)
            for k, v in obj.items()
        }
    return obj


def _replace_nodes_with_nodeables(obj, unwrap_stores=True):
    from meerkat.interactive.graph.store import Store

    if isinstance(obj, Node):
        obj = obj.obj
        if unwrap_stores:
            # Replace `Store` objects with their wrapped values
            if isinstance(obj, Store):
                obj = obj.__wrapped__
    elif isinstance(obj, list) or isinstance(obj, tuple):
        obj = type(obj)(_replace_nodes_with_nodeables(x) for x in obj)
    elif isinstance(obj, dict):
        obj = {
            _replace_nodes_with_nodeables(k): _replace_nodes_with_nodeables(v)
            for k, v in obj.items()
        }
    return obj


def _get_nodeables(*args, **kwargs) -> List[NodeMixin]:
    # TODO: figure out if we need to handle this case
    # Store([Store(1), Store(2), Store(3)])
    nodeables = []
    for arg in args:
        if isinstance(arg, NodeMixin):
            nodeables.append(arg)
        elif isinstance(arg, list) or isinstance(arg, tuple):
            nodeables.extend(_get_nodeables(*arg))
        elif isinstance(arg, dict):
            nodeables.extend(_get_nodeables(**arg))
        elif isinstance(arg, slice):
            nodeables.extend(_get_nodeables(arg.start, arg.stop, arg.step))

    for _, v in kwargs.items():
        if isinstance(v, NodeMixin):
            nodeables.append(v)
        elif isinstance(v, list) or isinstance(v, tuple):
            nodeables.extend(_get_nodeables(*v))
        elif isinstance(v, dict):
            nodeables.extend(_get_nodeables(**v))
        elif isinstance(v, slice):
            nodeables.extend(_get_nodeables(arg.start, arg.stop, arg.step))

    return nodeables
