from typing import ClassVar, Dict

from pydantic import BaseModel
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.node import Node, NodeMixin
from meerkat.interactive.graph import Store
from meerkat.mixins.identifiable import IdentifiableMixin


class ComponentConfig(BaseModel):
    component_id: str
    name: str
    props: Dict


class Component(IdentifiableMixin):

    _self_identifiable_group: str = "components"

    def __post_init__(self):
        """This is needed to support dataclasses on Components.
        https://docs.python.org/3/library/dataclasses.html#post-init-processing
        """
        super().__init__()

        # Custom setup
        self.setup()

        update_dict = {}
        for k, v in self.__dict__.items():
            new_v = v
            if k not in ["_self_id", "name", "identifiable_group"] and not isinstance(v, NodeMixin):
                # Convert literals to Store objects
                new_v = Store(v)
            elif k in ["_self_id", "name", "identifiable_group"]:
                continue

            # Now new_v is a NodeMixin object
            # We need to make sure that new_v points to a Node in the graph
            # If it doesn't, we need to add it to the graph
            if not new_v.has_inode():
                new_v.attach_to_inode(new_v.create_inode())
            
            # Now new_v is a NodeMixin object that points to a Node in the graph
            update_dict[k] = new_v.inode # this will exist

            assert isinstance(update_dict[k], Node)
        
        self.__dict__.update(update_dict)

    def setup(self):
        pass

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if isinstance(value, Node):
            return value.obj
        return value

    @property
    def config(self):
        return ComponentConfig(
            component_id=self.id, name=self.__class__.__name__, props=self.props
        )

    @property
    def _backend_only(self):
        return ["id", "name", "identifiable_group"]

    @property
    def props(self):
        props_dict = {}
        for k, v in self.__dict__.items():
            if k not in self._backend_only and v is not None:
                if hasattr(v, "config"):
                    if isinstance(v, Node) and (isinstance(v.obj, Store) or isinstance(v.obj, Endpoint)):
                        props_dict[k] = v.obj.config
                    else:
                        props_dict[k] = v.config
                else:
                    props_dict[k] = v
        return props_dict
