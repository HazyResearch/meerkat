from dataclasses import dataclass, field
from typing import Mapping, Sequence, Union
import uuid

from meerkat.interactive.frontend import FrontendMixin
from ..abstract import Component

@dataclass
class Tab(FrontendMixin):
    label: str
    component: Component
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def frontend(self):
        return {
            "id": self.id,
            "label": self.label,
            "component": self.component.frontend,
        }


class Tabs(Component):

    tabs: Union[Mapping[str, Component], Sequence[Tab]]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if isinstance(self.tabs, Mapping):
            self.tabs = [
                Tab(label=label, component=component)
                for label, component in self.tabs.items()
            ]
