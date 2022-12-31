from typing import TYPE_CHECKING, Literal, Union

from meerkat.interactive.app.src.lib.component.abstract import AutoComponent
from meerkat.interactive.app.src.lib.component.abstract import Slottable
from meerkat.mixins.identifiable import classproperty

if TYPE_CHECKING:
    from meerkat.interactive import Endpoint


class SvelteUIMixin:
    @classproperty
    def library(cls):
        return "@svelteuidev/core"

    @classproperty
    def namespace(cls):
        return "svelteui"


class Button(Slottable, SvelteUIMixin, AutoComponent):

    variant: str = "default"
    color: str = "indigo"
    gradient: dict = {}
    ripple: bool = False
    loading: bool = False
    loader_position: str = "left"
    radius: Union[str, int] = "md"
    size: Union[str, int] = "md"
    compact: bool = False
    full_size: bool = False
    href: str = None
    external: str = False
    disabled: bool = False
    upper_case: bool = False

    on_click: "Endpoint" = None
