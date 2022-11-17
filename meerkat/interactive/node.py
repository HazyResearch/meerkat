from collections import defaultdict
from typing import Dict, List, Optional

from pydantic import BaseModel

from meerkat.mixins.identifiable import IdentifiableMixin


class NodeConfig(BaseModel):
    ref_id: str
    type: str
    is_store: bool = True


class Node(IdentifiableMixin):

    _self_identifiable_group: str = "nodes"

    def __init__(self, obj: any, **kwargs):
        """ A node in the computational graph. This could be an object or an operation. 

        Args:
            obj (any): This could be any class that has NodeMixin (e.g. store, 
                Operation, DataFrame, Column). 
        """
        super().__init__(**kwargs)
        self.obj = obj
        self.children: Dict["Node", bool] = dict()

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
    def config(self):
        return NodeConfig(
            ref_id=self.id,
            type=self.obj.__class__.__name__,
        )

    @property
    def trigger_children(self):
        """Returns the children that are triggered."""
        return [child for child, triggers in self.children.items() if triggers]

    def __repr__(self) -> str:
        return f"Node({self.obj}, {len(self.children)} children)"

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


class NodeMixin:
    """
    Mixin for Classes whose objects can be attached to a node in the computation graph. 

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
        # self._self_children: Dict[Node, bool] = dict()
        self._self_inode = None  # Node(self)
        # self._set_node_id()

    @property
    def config(self):
        """Returns the config for the node."""
        # We used to assert if it wasn't already in the graph
        # assert (
        #     self.inode is not None
        # ), (
        #     "Something went wrong -- this object must be attached "
        #     "to a node in the graph."
        # )
        if self.inode is None:
            # to create a config that can be sent to the frontend 
            # the object must be attached to a node in the graph
            self.attach_to_inode(self.create_inode())

        return self.inode.config

    def attach_to_inode(self, inode: Node):
        """
        Attach this object to a node.
        """
        # The object should point to the node
        self._self_inode = inode
        # The node should point to the object
        inode.obj = self

    def detach_inode(self) -> Node:
        """
        Detach this object from its node.
        """
        # Grab the node
        inode = self._self_inode
        # Point the node to None
        inode.obj = None
        # The object should point to nothing
        self._self_inode = None
        # Return the node
        return inode

    def create_inode(self, inode_id: str = None) -> Node:
        """
        Creates a node for this object. Doesn't attach the
        node to the object yet.
        """
        return Node(None, id=inode_id)

    def has_inode(self):
        """Returns True if this object has a node."""
        return self._self_inode is not None

    @property
    def inode(self) -> Optional[Node]:
        """The node for this object, if it exists."""
        return self._self_inode

    @property
    def inode_id(self):
        return self.inode.id if self.inode else None

    def _set_inode(self):
        """Sets the node for this object."""
        self._self_inode = None


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
    # print("root_nodes", root_nodes)
    parents = defaultdict(set)
    nodes = set()
    while root_nodes:
        node = root_nodes.pop(0)
        # print("node", node, node.obj, node.children)
        for child in node.trigger_children:
            parents[child].add(node)
            nodes.add(node)
            root_nodes.append(child)

    current = [
        node for node in nodes if not parents[node]
    ]  # get a set of all the nodes without an incoming edge

    while current:
        node: Node = current.pop(0)
        yield node

        for child in node.trigger_children:
            parents[child].remove(node)
            if not parents[child]:
                current.append(child)
