from typing import Literal

from meerkat.interactive.app.src.lib.component.abstract import AutoComponent
from meerkat.interactive.app.src.lib.component.abstract import Slottable
from meerkat.interactive.endpoint import Endpoint
from meerkat.mixins.identifiable import classproperty


class CarbonComponentMixin:
    @classproperty
    def library(cls):
        return "carbon-components-svelte"

    @classproperty
    def namespace(cls):
        return "carbon"


class Accordion(CarbonComponentMixin, AutoComponent):

    align: Literal["start", "end"] = "end"
    size: Literal["sm", "xl"] = "sm"
    disabled: bool = False
    skeleton: bool = False


class AccordionItem(Slottable, CarbonComponentMixin, AutoComponent):

    open: bool = False
    disabled: bool = False
    title: str = "title"
    icon_description: str = "Expand/Collapse"

    on_animationend: Endpoint = None
    on_click: Endpoint = None
    on_keydown: Endpoint = None
    on_mouseleave: Endpoint = None
    on_mouseover: Endpoint = None


class AccordionSkeleton(CarbonComponentMixin, AutoComponent):

    count: int = 4
    align: Literal["start", "end"] = "end"
    size: Literal["sm", "xl"] = "sm"
    open: bool = True

    on_click: Endpoint = None
    on_mouseenter: Endpoint = None
    on_mouseleave: Endpoint = None
    on_mouseover: Endpoint = None


class Button(Slottable, CarbonComponentMixin, AutoComponent):

    kind: Literal[
        "primary",
        "secondary",
        "tertiary",
        "ghost",
        "danger",
        "danger-tertiary",
        "danger-ghost",
    ] = "primary"

    on_click: Endpoint = None
    on_mouseenter: Endpoint = None
    on_mouseleave: Endpoint = None
    on_mouseover: Endpoint = None


__all__ = [
    "Accordion",
    "AccordionItem",
    "AccordionSkeleton",
    "Button",
]
