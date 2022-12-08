import inspect
import os
from typing import Dict

from pydantic import BaseModel, Extra, validator

from meerkat.interactive.frontend import FrontendMixin
from meerkat.interactive.graph import Store
from meerkat.interactive.node import Node, NodeMixin
from meerkat.mixins.identifiable import IdentifiableMixin, classproperty
from meerkat.tools.utils import nested_apply


class ComponentFrontend(BaseModel):
    component_id: str
    path: str
    name: str
    props: Dict


# need to pass the extra param in order to
class Component(IdentifiableMixin, FrontendMixin, BaseModel):
    @classproperty
    def identifiable_group(self):
        # Ordinarily, we would create a new classproperty for this, like
        # _self_identifiable_group: str = "components"
        # However, this causes pydantic to show _self_identifiable_group in
        # type hints when using the component in the IDE, which might
        # be confusing to users.
        # We just override the classproperty here directly as an alternative.
        return "components"

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
    def props(self):
        return {k: self.__getattribute__(k) for k in self.__fields__ if "_self_id" != k}

    @property
    def frontend(self):
        def _frontend(value):
            if isinstance(value, FrontendMixin):
                return value.frontend
            return value

        frontend_props = nested_apply(
            self.props,
            _frontend,
            base_types=(Store),
        )

        component_name = self.__class__.__name__
        # Inheriting an existing Component and modifying it on the Python side
        # should not change the name of the component used on the frontend
        if self.__class__.__bases__[0] != Component and issubclass(
            self.__class__.__bases__[0], Component
        ):
            component_name = self.__class__.__bases__[0].__name__

        return ComponentFrontend(
            component_id=self.id,
            path=os.path.join(
                os.path.dirname(inspect.getfile(self.__class__)),
                f"{component_name}.svelte",
            ),
            name=component_name,
            props=frontend_props,
        )

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow
        copy_on_model_validation = False
