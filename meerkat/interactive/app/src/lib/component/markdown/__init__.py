from dataclasses import dataclass

from ..abstract import Component


class Markdown(Component):

    name = "Markdown"

    def __init__(
        self,
        value: str,
    ) -> None:
        super().__init__()
        self.value = value

    @property
    def props(self):
        props = {
            "value": self.value,
        }
        return props
