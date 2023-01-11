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


def _replace_nodes_with_nodeables(obj):
    if isinstance(obj, Node):
        obj = obj.obj
    elif isinstance(obj, list) or isinstance(obj, tuple):
        obj = type(obj)(_replace_nodes_with_nodeables(x) for x in obj)
    elif isinstance(obj, dict):
        obj = {
            _replace_nodes_with_nodeables(k): _replace_nodes_with_nodeables(v)
            for k, v in obj.items()
        }
    return obj


def _get_nodeables(*args, **kwargs):
    nodeables = []
    for arg in args:
        if isinstance(arg, NodeMixin):
            nodeables.append(arg)
        elif isinstance(arg, list) or isinstance(arg, tuple):
            nodeables.extend(_get_nodeables(*arg))
        elif isinstance(arg, dict):
            nodeables.extend(_get_nodeables(**arg))

    for _, v in kwargs.items():
        if isinstance(v, NodeMixin):
            nodeables.append(v)
        elif isinstance(v, list) or isinstance(v, tuple):
            nodeables.extend(_get_nodeables(*v))
        elif isinstance(v, dict):
            nodeables.extend(_get_nodeables(**v))
    return nodeables
