from dataclasses import dataclass, field
from typing import Mapping, Sequence, Union
import uuid
from ..abstract import Component


@dataclass
class Tab:
    label: str
    component: Component
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def config(self):
        return {
            "id": self.id,
            "label": self.label,
            "component": self.component.config,
        }


@dataclass
class Tabs(Component):

    tabs: Union[Mapping[str, Component], Sequence[Tab]]

    def __post_init__(self):
        if isinstance(self.tabs, Mapping):
            self.tabs = [
                Tab(label=label, component=component)
                for label, component in self.tabs.items()
            ]
        super().__post_init__()
