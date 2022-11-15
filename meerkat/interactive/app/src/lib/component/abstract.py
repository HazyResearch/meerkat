from typing import Dict

from pydantic import BaseModel

from meerkat.mixins.identifiable import IdentifiableMixin


class ComponentConfig(BaseModel):
    component_id: str
    name: str
    props: Dict


class Component(IdentifiableMixin):

    _self_identifiable_group: str = "components"

    name: str

    @property
    def config(self):
        return ComponentConfig(component_id=self.id, name=self.name, props=self.props)

    @property
    def props(self):
        return {
            # TODO: improve this so we isinstance a class instead
            k: v.config if hasattr(v, "config") else v
            for k, v in self.__dict__.items()
            # FIXME: critical fix, need to remove all keys here
            if k not in ["id", "name", "identifiable_group"]
        }
