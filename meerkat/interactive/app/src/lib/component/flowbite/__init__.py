from typing import Literal

from meerkat.interactive.app.src.lib.component.abstract import AutoComponent, Slottable
from meerkat.interactive.endpoint import Endpoint
from meerkat.mixins.identifiable import classproperty


class FlowbiteSvelteMixin:
    @classproperty
    def library(cls):
        return "flowbite-svelte"

    @classproperty
    def namespace(cls):
        return "flowbite"


# class Accordion(Slottable, FlowbiteSvelteMixin, AutoComponent):
#     multiple: bool = False
#     flush: bool = False
#     activeClasses: str = "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-4 focus:ring-gray-200 dark:focus:ring-gray-800"
#     inactiveClasses: str = (
#         "text-gray-500 dark:text-gray-400 hover:bg-gray-100 hover:dark:bg-gray-800"
#     )
#     defaultClass: str = "text-gray-500 dark:text-gray-400"


# class AccordionItem(Slottable, FlowbiteSvelteMixin, AutoComponent):
#     open: bool = False
#     activeClasses: str = None
#     inactiveClasses: str = None
#     defaultClass: str = "flex items-center justify-between w-full font-medium text-left group-first:rounded-t-xl"
#     transitionType: Literal["slide", "fade"] = "slide"
#     transitionParams: dict = {}


# class Button(Slottable, FlowbiteSvelteMixin, AutoComponent):

#     pill: bool = False
#     outline: bool = False
#     gradient: bool = False
#     size: Literal["xs", "sm", "md", "lg", "xl"] = "md"
#     href: str = None
#     btnClass: str = None
#     type: Literal["button", "submit", "reset"] = "button"
#     color: Literal[
#         "alternative",
#         "blue",
#         "cyan",
#         "dark",
#         "light",
#         "lime",
#         "green",
#         "pink",
#         "primary",
#         "red",
#         "teal",
#         "yellow",
#         "purple",
#         "purpleToBlue",
#         "cyanToBlue",
#         "greenToBlue",
#         "purpleToPink",
#         "pinkToOrange",
#         "tealToLime",
#         "redToYellow",
#     ] = "blue"
#     shadow: Literal[
#         "blue", "green", "cyan", "teal", "lime", "red", "pink", "purple"
#     ] = None

#     on_change: Endpoint = None
#     on_click: Endpoint = None
#     on_keydown: Endpoint = None
#     on_keyup: Endpoint = None
#     on_mouseenter: Endpoint = None
#     on_mouseleave: Endpoint = None


# class Card(Slottable, FlowbiteSvelteMixin, AutoComponent):

#     href: str = None
#     horizontal: bool = False
#     reverse: bool = False
#     img: str = None
#     padding: Literal["none", "sm", "md", "lg", "xl"] = "lg"
#     size: Literal["xs", "sm", "md", "lg", "xl"] = "sm"

#     # Frame options, passed as $$restProps
#     tag: Literal["div", "a"] = "div"
#     color: Literal[
#         "gray",
#         "red",
#         "yellow",
#         "green",
#         "indigo",
#         "default",
#         "purple",
#         "pink",
#         "blue",
#         "light",
#         "dark",
#         "dropdown",
#         "navbar",
#         "navbarUl",
#         "form",
#         "none",
#     ] = "default"
#     rounded: bool = False
#     border: bool = False
#     shadow: bool = False


# class Dropdown(Slottable, FlowbiteSvelteMixin, AutoComponent):

#     open: bool = False
#     frameClass: str = ""

#     # Configure slots for header and footer


# class DropdownItem(Slottable, FlowbiteSvelteMixin, AutoComponent):

#     defaultClass: str = (
#         "font-medium py-2 px-4 text-sm hover:bg-gray-100 dark:hover:bg-gray-600"
#     )
#     href: str = None

#     # Events
#     on_blur: Endpoint = None
#     on_change: Endpoint = None
#     on_click: Endpoint = None
#     on_focus: Endpoint = None
#     on_keydown: Endpoint = None
#     on_keyup: Endpoint = None
#     on_mouseenter: Endpoint = None
#     on_mouseleave: Endpoint = None


# class DropdownDivider(Slottable, FlowbiteSvelteMixin, AutoComponent):

#     divClass: str = "my-1 h-px bg-gray-100 dark:bg-gray-600"


# class DropdownHeader(Slottable, FlowbiteSvelteMixin, AutoComponent):

#     divClass: str = "py-2 px-4 text-gray-700 dark:text-white"
#     divider: bool = True


# __all__ = [
#     "Button",
#     "Card",
#     "Dropdown",
#     "DropdownItem",
#     "DropdownDivider",
#     "DropdownHeader",
# ]
