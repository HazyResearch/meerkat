import uuid
from dataclasses import dataclass, field
from typing import Mapping, Sequence, Union

from meerkat.interactive.app.src.lib.component.abstract import BaseComponent
from meerkat.interactive.frontend import FrontendMixin


@dataclass
class Tab(FrontendMixin):
    label: str
    component: BaseComponent
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def frontend(self):
        return {
            "id": self.id,
            "label": self.label,
            "component": self.component.frontend,
        }


class Tabs(BaseComponent):
    # TODO: Add option for setting the default selected tab.
    tabs: Union[Mapping[str, BaseComponent], Sequence[Tab]]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if isinstance(self.tabs, Mapping):
            self.tabs = [
                Tab(label=label, component=component)
                for label, component in self.tabs.items()
            ]
