from ..app.src.lib.component.core.icon import Icon
from .base import Formatter


class IconFormatter(Formatter):
    component_class = Icon
    data_prop: str = "data"
    static_encode: bool = True

    def encode(self, data: str) -> str:
        return ""
