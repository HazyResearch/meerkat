from typing import Dict

from pydantic import BaseModel
from meerkat.interactive.node import NodeMixin
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

        self.__dict__.update(
            {
                # TODO: improve this so we isinstance a class instead
                k: v if isinstance(v, NodeMixin) else Store(v)
                for k, v in self.__dict__.items()
                # FIXME: critical fix, need to remove all keys here
                if (
                    k not in ["_self_id", "name", "identifiable_group"]
                    and not isinstance(v, NodeMixin)
                )
            }
        )

    @property
    def config(self):
        return ComponentConfig(
            component_id=self.id, name=self.__class__.__name__, props=self.props
        )

    @property
    def props(self):
        return {
            # TODO: improve this so we isinstance a class instead
            k: v.config if hasattr(v, "config") else v
            for k, v in self.__dict__.items()
            # FIXME: critical fix, need to remove all keys here
            if k not in ["id", "name", "identifiable_group"] and v is not None
        }
