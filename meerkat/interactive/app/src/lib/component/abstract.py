from typing import Dict

from pydantic import BaseModel, Extra, validator
from meerkat.interactive.node import Node, NodeMixin
from meerkat.interactive.graph import Store
from meerkat.interactive.frontend import FrontendMixin
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.tools.utils import nested_apply


class ComponentFrontend(BaseModel):
    component_id: str
    name: str
    props: Dict


# need to pass the extra param in order to
class Component(IdentifiableMixin, FrontendMixin, BaseModel):

    """Create

    Returns:
        _type_: _description_
    """

    _self_identifiable_group: str = "components"

    @validator("*", pre=False)
    def check_inode(cls, value):
        if isinstance(value, NodeMixin) and not isinstance(value, Store):
            # Now value is a NodeMixin object
            # We need to make sure that value points to a Node in the graph
            # If it doesn't, we need to add it to the graph
            if not value.has_inode():
                value.attach_to_inode(value.create_inode())

            # Now value is a NodeMixin object that points to a Node in the graph
            return value.inode  # this will exist
        return value

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if isinstance(value, Node):
            # because the validator above converts dataframes to nodes, when the
            # dataframe is accessed we need to convert it back to the dataframe
            return value.obj
        return value

    @property
    def frontend(self):
        def _frontend(value):
            if isinstance(value, FrontendMixin):
                return value.frontend
            return value

        frontend_props = nested_apply(
            {k: self.__getattribute__(k) for k in self.__fields__ if "_self_id" != k},
            _frontend,
            base_types=(Store)
        )
        return ComponentFrontend(
            component_id=self.id, name=self.__class__.__name__, props=frontend_props
        )

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow
