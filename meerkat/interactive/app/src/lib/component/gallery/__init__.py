from meerkat.interactive import Box

from ..abstract import Component


class Gallery(Component):

    name = "Gallery"

    def __init__(
        self,
        dp: Box,
    ) -> None:
        super().__init__()
        self.dp = dp

    @property
    def props(self):
        return {
            "dp": self.dp.config,
        }
