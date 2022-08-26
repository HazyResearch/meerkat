from typing import Dict

from pydantic import BaseModel

from meerkat.mixins.identifiable import IdentifiableMixin


class ComponentConfig(BaseModel):
    component_id: str
    component: str
    props: Dict


class Component(IdentifiableMixin):

    identifiable_group: str = "components"

    name: str

    @property
    def config(self):
        return ComponentConfig(
            component_id=self.id, component=self.name, props=self.props
        )

    @property
    def props(self):
        return {}
