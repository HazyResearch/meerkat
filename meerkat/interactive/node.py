from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from meerkat.interactive.graph import Operation


class NodeMixin:
    """
    Class for defining nodes in a graph.

    Add this mixin to any class whose objects should be nodes
    in a graph.

    This mixin is used in Reference, Store and Operation to make
    them part of a computation graph.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The children of this node: this is a dictionary
        # mapping children to a boolean indicating whether
        # the child is triggered when this node is triggered.
        self.children: Dict[Operation, bool] = dict()

    def add_child(self, child, triggers=True):
        """Adds a child to this node.

        Args:
            child: The child to add.
            triggers: If True, this child is triggered
                when this node is triggered.
        """
        if child not in self.children:
            self.children[child] = triggers

        # Don't overwrite triggers=True with triggers=False
        self.children[child] = triggers | self.children[child]

    @property
    def trigger_children(self):
        """Returns the children that are triggered."""
        return [child for child, triggers in self.children.items() if triggers]

    def __hash__(self):
        """Hash is based on the id of the node."""
        return hash(id(self))

    def __eq__(self, other):
        """Two nodes are equal if they have the same id."""
        return id(self) == id(other)

    def has_children(self):
        """Returns True if this node has children."""
        return len(self.children) > 0

    def has_trigger_children(self):
        """Returns True if this node has children that are triggered."""
        return any(self.children.values())


def _topological_sort(root_nodes: List[NodeMixin]) -> List[NodeMixin]:
    """
    Perform a topological sort on a graph.

    Args:
        root_nodes (List[NodeMixin]): The root nodes of the graph.

    Returns:
        List[NodeMixin]: The topologically sorted nodes.
    """
    # get a mapping from node to the children of each node
    # only get the children that are triggered by the node
    # i.e. ignore children that use the node as a dependency
    # but are not triggered by the node
    parents = defaultdict(set)
    nodes = set()
    while root_nodes:
        node = root_nodes.pop(0)
        for child in node.trigger_children:
            parents[child].add(node)
            nodes.add(node)
            root_nodes.append(child)

    current = [
        node for node in nodes if not parents[node]
    ]  # get a set of all the nodes without an incoming edge

    while current:
        node: NodeMixin = current.pop(0)
        yield node

        for child in node.trigger_children:
            parents[child].remove(node)
            if not parents[child]:
                current.append(child)
