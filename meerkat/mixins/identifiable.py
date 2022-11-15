from uuid import uuid4


class classproperty(property):
    """
    Taken from https://stackoverflow.com/a/13624858

    The behavior of class properties using the @classmethod
    and @property decorators has changed across Python versions.
    This class (should) provide consistent behavior across Python
    versions.
    See https://stackoverflow.com/a/1800999 for more information.
    """

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class Graph:
    def __init__(self):
        self._adjacency_list = dict()

    def add_node(self, node):
        self._adjacency_list[node] = set()

    def add_edge(self, node1, node2):
        self._adjacency_list[node1].add(node2)

    def get_adjacency_list(self):
        return self._adjacency_list

    @property
    def nodes(self):
        return list(self._adjacency_list.keys())

    @property
    def edges(self):
        edges = []
        for node, children in self._adjacency_list.items():
            for child in children:
                edges.append((node, child))
        return edges

    def get_children(self, node):
        return self._adjacency_list[node]

    def get_parents(self, node):
        parents = []
        for parent, children in self._adjacency_list.items():
            if node in children:
                parents.append(parent)
        return parents

    @property
    def root_nodes(self):
        root_nodes = []
        for node in self._adjacency_list.keys():
            if not self.get_parents(node):
                root_nodes.append(node)
        return root_nodes

    @property
    def leaf_nodes(self):
        leaf_nodes = []
        for node in self._adjacency_list.keys():
            if not self.get_children(node):
                leaf_nodes.append(node)
        return leaf_nodes


class GraphMixin:
    def __init__(self, *args, **kwargs):
        super(GraphMixin, self).__init__(*args, **kwargs)
        self._self_node_id = uuid4().hex

    @property
    def node_id(self):
        return self._self_node_id

    @node_id.setter
    def node_id(self, value):
        self._self_node_id = value


class IdentifiableMixin:
    """
    Mixin for classes, to give objects an id.

    This class must use _self_{attribute} for all attributes
    since it will be mixed into the wrapt.ObjectProxy class,
    which requires this naming convention for it to work.
    """

    _self_identifiable_group: str

    def __init__(self, id: str = None, *args, **kwargs):
        super(IdentifiableMixin, self).__init__(*args, **kwargs)
        self._set_id(id=id)

    @property
    def id(self):
        return self._self_id

    # Note: this is set to be a classproperty, so that we can access either
    # cls.identifiable_group or self.identifiable_group.

    # If this is changed to a property, then we can only access
    # self.identifiable_group, and cls._self_identifiable_group but not
    # cls.identifiable_group. This is fine if something breaks, but
    # be careful to change cls.identifiable_group to
    # cls._self_identifiable_group everywhere.
    @classproperty
    def identifiable_group(self):
        return self._self_identifiable_group

    def _set_id(self, id: str = None):
        # get uuid as str
        if id is None:
            self._self_id = uuid4().hex
        else:
            self._self_id = id

        from meerkat.state import state

        state.identifiables.add(self)

    @classmethod
    def from_id(cls, id: str):
        # TODO(karan): make sure we're using this everywhere and it's not
        # being redefined in subclasses
        from meerkat.state import state

        return state.identifiables.get(id=id, group=cls._self_identifiable_group)
