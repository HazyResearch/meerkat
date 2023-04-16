from pydantic import BaseModel, Extra, validator

from meerkat.interactive.endpoint import endpoint
from meerkat.interactive.graph import Store
from meerkat.interactive.node import Node, NodeMixin
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.tools.utils import classproperty


class EndpointMixin:
    def __init__(self, prefix=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._endpoints = {}

        if prefix is None:
            prefix = f"/{self.__class__.__name__.lower()}"

        # Access all the user-defined attributes of the instance to create endpoints
        # Here, we keep only methods that:
        # - are defined in this subclass, but not in any of its superclasses
        #   (e.g. BaseComponent, IdentifiableMixin, EndpointMixin etc.)
        # - don't begin with "_"
        # - are callables
        names = set(dir(self.__class__)) - set(
            sum([dir(e) for e in self.__class__.mro()[1:]], [])
        )
        for attrib in names:
            if attrib.startswith("_"):
                continue
            obj = self.__getattribute__(attrib)
            if callable(obj):
                if attrib not in self._endpoints:
                    print(attrib)
                    self._endpoints[attrib] = endpoint(
                        obj, prefix=prefix + f"/{self.id}"
                    )

    @property
    def endpoints(self):
        return self._endpoints


class State(EndpointMixin, IdentifiableMixin, BaseModel):
    @classproperty
    def identifiable_group(self):
        # Ordinarily, we would create a new classproperty for this, like
        # _self_identifiable_group: str = "states"
        # However, this causes pydantic to show _self_identifiable_group in
        # type hints when using the component in the IDE, which might
        # be confusing to users.
        # We just override the classproperty here directly as an alternative.
        return "states"

    @validator("*", pre=False)
    def _check_inode(cls, value):
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
        if callable(value) and hasattr(self, "_endpoints"):
            if name not in self._endpoints:
                return value
            return self._endpoints[name]
        return value

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow
        copy_on_model_validation = False
