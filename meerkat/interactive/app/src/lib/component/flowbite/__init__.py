import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from meerkat.interactive.app.src.lib.component.abstract import Component, Slottable
from meerkat.interactive.endpoint import Endpoint
from meerkat.tools.utils import classproperty

from .types import (
    ActivityType,
    Buttonshadows,
    ButtonType,
    DrawerTransitionTypes,
    FormColorType,
    GroupTimelineType,
    LinkType,
    Placement,
    ReviewType,
    SiteType,
    SizeType,
    TransitionTypes,
    number,
)


class FlowbiteSvelteMixin:
    @classproperty
    def library(cls):
        return "flowbite-svelte"

    @classproperty
    def namespace(cls):
        return "flowbite"


class Accordion(Slottable, FlowbiteSvelteMixin, Component):
    multiple: bool = False
    flush: bool = False
    activeClasses: str = "bg-gray-100 dark:bg-gray-800 text-gray-900 "
    "dark:text-white focus:ring-4 focus:ring-gray-200 dark:focus:ring-gray-800"
    inactiveClasses: str = (
        "text-gray-500 dark:text-gray-400 hover:bg-gray-100 hover:dark:bg-gray-800"
    )
    defaultClass: str = "text-gray-500 dark:text-gray-400"


class AccordionItem(Slottable, FlowbiteSvelteMixin, Component):
    open: bool = False
    activeClasses: str = None
    inactiveClasses: str = None
    defaultClass: str = "flex items-center justify-between w-full font-medium "
    "text-left group-first:rounded-t-xl"
    transitionType: Literal["slide", "fade"] = "slide"
    transitionParams: dict = {}


class Alert(Slottable, FlowbiteSvelteMixin, Component):
    dismissable: bool = False
    accent: bool = False
    color: Literal[
        "blue", "dark", "red", "green", "yellow", "indigo", "purple", "pink"
    ] = "blue"


class Avatar(Slottable, FlowbiteSvelteMixin, Component):
    src: str = ""
    href: str = None
    rounded: bool = False
    border: bool = False
    stacked: bool = False
    dot: dict = None
    alt: str = ""
    size: SizeType = "md"

    classes: str = None


class Badge(Slottable, FlowbiteSvelteMixin, Component):
    color: Literal[
        "blue", "dark", "red", "green", "yellow", "indigo", "purple", "pink"
    ] = "blue"
    large: bool = False
    border: bool = False
    href: str = None
    rounded: bool = False
    index: bool = False
    dismissable: bool = False


class Breadcrumb(Slottable, FlowbiteSvelteMixin, Component):
    solid: bool = False
    navClass: str = "flex"
    solidClass: str = "flex px-5 py-3 text-gray-700 border border-gray-200 "
    "rounded-lg bg-gray-50 dark:bg-gray-800 dark:border-gray-700"
    olClass: str = "inline-flex items-center space-x-1 md:space-x-3"


class BreadcrumbItem(Slottable, FlowbiteSvelteMixin, Component):
    home: bool = False
    href: str = None
    linkClass: str = "ml-1 text-sm font-medium text-gray-700 hover:text-gray-900 "
    "md:ml-2 dark:text-gray-400 dark:hover:text-white"
    spanClass: str = "ml-1 text-sm font-medium text-gray-500 md:ml-2 "
    "dark:text-gray-400"
    homeClass: str = "inline-flex items-center text-sm font-medium "
    "text-gray-700 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"


class Button(Slottable, FlowbiteSvelteMixin, Component):
    classes: str = None

    pill: bool = False
    outline: bool = False
    gradient: bool = False
    size: SizeType = "md"
    href: str = None
    btnClass: str = None
    type: ButtonType = "button"
    color: Literal[
        "alternative",
        "blue",
        "cyan",
        "dark",
        "light",
        "lime",
        "green",
        "pink",
        "primary",
        "red",
        "teal",
        "yellow",
        "purple",
        "purpleToBlue",
        "cyanToBlue",
        "greenToBlue",
        "purpleToPink",
        "pinkToOrange",
        "tealToLime",
        "redToYellow",
    ] = "blue"
    shadow: Optional[Buttonshadows] = None

    on_change: Endpoint = None
    on_click: Endpoint = None
    on_keydown: Endpoint = None
    on_keyup: Endpoint = None
    on_mouseenter: Endpoint = None
    on_mouseleave: Endpoint = None


class ButtonGroup(Slottable, FlowbiteSvelteMixin, Component):
    size: SizeType = "md"
    divClass: str = "inline-flex rounded-lg shadow-sm"

    on_blur: Endpoint = None
    on_change: Endpoint = None
    on_click: Endpoint = None
    on_focus: Endpoint = None
    on_keydown: Endpoint = None
    on_keyup: Endpoint = None
    on_mouseenter: Endpoint = None
    on_mouseleave: Endpoint = None


class Card(Slottable, FlowbiteSvelteMixin, Component):
    href: str = None
    horizontal: bool = False
    reverse: bool = False
    img: str = None
    padding: Literal["none", "sm", "md", "lg", "xl"] = "lg"
    size: SizeType = "sm"

    # Frame options, passed as $$restProps
    tag: Literal["div", "a"] = "div"
    color: Literal[
        "gray",
        "red",
        "yellow",
        "green",
        "indigo",
        "default",
        "purple",
        "pink",
        "blue",
        "light",
        "dark",
        "dropdown",
        "navbar",
        "navbarUl",
        "form",
        "none",
    ] = "default"
    rounded: bool = False
    border: bool = False
    shadow: bool = False


class Carousel(Slottable, FlowbiteSvelteMixin, Component):
    showIndicators: bool = True
    showCaptions: bool = True
    showThumbs: bool = True
    images: List[Dict[str, Any]] = []
    slideControls: bool = True
    loop: bool = False
    duration: int = 2000
    divClass: str = "overflow-hidden relative h-56 rounded-lg sm:h-64 xl:h-80 2xl:h-96"
    indicatorDivClass: str = (
        "flex absolute bottom-5 left-1/2 z-30 space-x-3 -translate-x-1/2"
    )
    captionClass: str = (
        "h-10 bg-gray-300 dark:bg-gray-700 dark:text-white p-2 my-2 text-center"
    )
    indicatorClass: str = (
        "w-3 h-3 rounded-full bg-gray-100 hover:bg-gray-300 opacity-60"
    )
    slideClass: str = ""


class CarouselTransition(Slottable, FlowbiteSvelteMixin, Component):
    showIndicators: bool = True
    showCaptions: bool = True
    showThumbs: bool = True
    images: List[Dict[str, Any]] = []
    slideControls: bool = True
    transitionType: TransitionTypes = "fade"
    transitionParams: dict = {}  # TransitionParamTypes
    loop: bool = False
    duration: int = 2000
    divClass: str = "overflow-hidden relative h-56 rounded-lg sm:h-64 xl:h-80 2xl:h-96"
    indicatorDivClass: str = (
        "flex absolute bottom-5 left-1/2 z-30 space-x-3 -translate-x-1/2"
    )
    captionClass: str = (
        "h-10 bg-gray-300 dark:bg-gray-700 dark:text-white p-2 my-2 text-center"
    )
    indicatorClass: str = (
        "w-3 h-3 rounded-full bg-gray-100 hover:bg-gray-300 opacity-60"
    )


class DarkMode(Slottable, FlowbiteSvelteMixin, Component):
    btnClass: str = (
        "text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 "
        "focus:outline-none focus:ring-4 focus:ring-gray-200 dark:focus:ring-gray-700 "
        "rounded-lg text-sm p-2.5"
    )


class Drawer(Slottable, FlowbiteSvelteMixin, Component):
    activateClickOutside: bool = True
    hidden: bool = True
    position: Literal["fixed", "absolute"] = "fixed"
    leftOffset: str = "inset-y-0 left-0"
    rightOffset: str = "inset-y-0 right-0"
    topOffset: str = "inset-x-0 top-0"
    bottomOffset: str = "inset-x-0 bottom-0"
    width: str = "w-80"
    backdrop: bool = True
    bgColor: str = "bg-gray-900"
    bgOpacity: str = "bg-opacity-75"
    placement: Literal["left", "right", "top", "bottom"] = "left"
    # id: str = "drawer-example"
    divClass: str = "overflow-y-auto z-50 p-4 bg-white dark:bg-gray-800"
    transitionParams: dict = {}  # DrawerTransitionParamTypes
    transitionType: DrawerTransitionTypes = "fly"


class Dropdown(Slottable, FlowbiteSvelteMixin, Component):
    open: bool = False
    frameClass: str = ""

    # Configure slots for header and footer


class DropdownItem(Slottable, FlowbiteSvelteMixin, Component):
    defaultClass: str = (
        "font-medium py-2 px-4 text-sm hover:bg-gray-100 dark:hover:bg-gray-600"
    )
    href: str = None

    # Events
    on_blur: Endpoint = None
    on_change: Endpoint = None
    on_click: Endpoint = None
    on_focus: Endpoint = None
    on_keydown: Endpoint = None
    on_keyup: Endpoint = None
    on_mouseenter: Endpoint = None
    on_mouseleave: Endpoint = None


class DropdownDivider(Slottable, FlowbiteSvelteMixin, Component):
    divClass: str = "my-1 h-px bg-gray-100 dark:bg-gray-600"


class DropdownHeader(Slottable, FlowbiteSvelteMixin, Component):
    divClass: str = "py-2 px-4 text-gray-700 dark:text-white"
    divider: bool = True


class Footer(Slottable, FlowbiteSvelteMixin, Component):
    customClass: str = ""
    footerType: Literal[
        "custom", "sitemap", "default", "logo", "socialmedia"
    ] = "default"


class FooterBrand(Slottable, FlowbiteSvelteMixin, Component):
    aClass: str = "flex items-center"
    spanClass: str = (
        "self-center text-2xl font-semibold whitespace-nowrap dark:text-white"
    )
    imgClass: str = "mr-3 h-8"
    href: str = ""
    src: str = ""
    alt: str = ""
    name: str = ""
    target: str = None


class FooterCopyright(Slottable, FlowbiteSvelteMixin, Component):
    spanClass: str = "block text-sm text-gray-500 sm:text-center dark:text-gray-400"
    aClass: str = "hover:underline"
    year: number = int(datetime.date.today().strftime("%Y"))  # current year
    href: str = ""
    by: str = ""
    target: str = None


class FooterIcon(Slottable, FlowbiteSvelteMixin, Component):
    href: str = ""
    ariaLabel: str = ""
    aClass: str = "text-gray-500 hover:text-gray-900 dark:hover:text-white"
    target: str = None


class FooterLink(Slottable, FlowbiteSvelteMixin, Component):
    liClass: str = "mr-4 last:mr-0 md:mr-6"
    aClass: str = "hover:underline"
    href: str = ""
    target: str = None


class FooterLinkGroup(Slottable, FlowbiteSvelteMixin, Component):
    ulClass: str = "text-gray-600 dark:text-gray-400"


class Indicator(Slottable, FlowbiteSvelteMixin, Component):
    color: Literal[
        "gray",
        "dark",
        "blue",
        "green",
        "red",
        "purple",
        "indigo",
        "yellow",
        "teal",
        "none",
    ] = "gray"
    rounded: bool = False
    size: SizeType = "md"
    border: bool = False
    placement: Literal[
        "top-left",
        "top-center",
        "top-right",
        "center-left",
        "center",
        "center-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    ] = None
    offset: bool = True


class Kbd(Slottable, FlowbiteSvelteMixin, Component):
    kbdClass: str = "text-xs font-semibold text-gray-800 bg-gray-100 "
    "border border-gray-200 rounded-lg dark:bg-gray-600 "
    "dark:text-gray-100 dark:border-gray-500"


class ArrowKeyDown(Slottable, FlowbiteSvelteMixin, Component):
    svgClass: str = "w-4 h-4"


class ArrowKeyLeft(Slottable, FlowbiteSvelteMixin, Component):
    svgClass: str = "w-4 h-4"


class ArrowKeyRight(Slottable, FlowbiteSvelteMixin, Component):
    svgClass: str = "w-4 h-4"


class ArrowKeyUp(Slottable, FlowbiteSvelteMixin, Component):
    svgClass: str = "w-4 h-4"


# class ListGroup(Slottable, FlowbiteSvelteMixin, Component):

#     items: List[ListGroupItemType] = []
#     active: bool = False
#     classes: str = ""


# class ListGroupItem(Slottable, FlowbiteSvelteMixin, Component):

#     classes: str = ""

#     # Events
#     on_blur: Endpoint = None
#     on_change: Endpoint = None
#     on_click: Endpoint = None
#     on_focus: Endpoint = None
#     on_keydown: Endpoint = None
#     on_keypress: Endpoint = None
#     on_keyup: Endpoint = None
#     on_mouseenter: Endpoint = None
#     on_mouseleave: Endpoint = None
#     on_mouseover: Endpoint = None


class MegaMenu(Slottable, FlowbiteSvelteMixin, Component):
    classes: str = None
    items: List[LinkType] = []
    open: bool = False
    full: bool = False


class Modal(Slottable, FlowbiteSvelteMixin, Component):
    open: bool = False
    title: str = ""
    size: SizeType = "md"
    placement: Literal[
        "top-left",
        "top-center",
        "top-right",
        "center-left",
        "center",
        "center-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    ] = "center"
    autoclose: bool = False
    permanent: bool = False
    backdropClasses: str = "bg-gray-900 bg-opacity-50 dark:bg-opacity-80"

    on_hide: Endpoint = None
    on_open: Endpoint = None


class Navbar(Slottable, FlowbiteSvelteMixin, Component):
    navClass: str = "px-2 sm:px-4 py-2.5 w-full"
    navDivClass: str = "mx-auto flex flex-wrap justify-between items-center "
    fluid: bool = True
    color: Literal[
        "gray",
        "red",
        "yellow",
        "green",
        "indigo",
        "default",
        "purple",
        "pink",
        "blue",
        "light",
        "dark",
        "dropdown",
        "navbar",
        "navbarUl",
        "form",
        "none",
    ] = "navbar"


class NavBrand(Slottable, FlowbiteSvelteMixin, Component):
    href: str = ""


class NavLi(Slottable, FlowbiteSvelteMixin, Component):
    href: str = ""
    active: bool = False
    activeClass: str = "text-white bg-blue-700 md:bg-transparent "
    "md:text-blue-700 md:dark:text-white dark:bg-blue-600 md:dark:bg-transparent"
    nonActiveClass: str = "text-gray-700 hover:bg-gray-100 md:hover:bg-transparent "
    "md:border-0 md:hover:text-blue-700 dark:text-gray-400 md:dark:hover:text-white "
    "dark:hover:bg-gray-700 dark:hover:text-white md:dark:hover:bg-transparent"


class NavUl(Slottable, FlowbiteSvelteMixin, Component):
    divClass: str = "w-full md:block md:w-auto"
    ulClass: str = "flex flex-col p-4 mt-4 md:flex-row md:space-x-8 md:mt-0 "
    "md:text-sm md:font-medium"
    hidden: bool = True
    # slideParams: SlideParams = {delay: 250, duration: 500, easing: quintOut}


class Pagination(Slottable, FlowbiteSvelteMixin, Component):
    pages: List[LinkType] = []
    activeClass: str = "text-blue-600 border border-gray-300 bg-blue-50 "
    "hover:bg-blue-100 hover:text-blue-700 dark:border-gray-700 "
    "dark:bg-gray-700 dark:text-white"
    normalClass: str = "text-gray-500 bg-white hover:bg-gray-100 "
    "hover:text-gray-700 dark:bg-gray-800 dark:border-gray-700 "
    "dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white"
    ulClass: str = "inline-flex -space-x-px items-center"
    table: bool = False


class PaginationItem(Slottable, FlowbiteSvelteMixin, Component):
    href: str = None
    active: bool = False
    activeClass: str = ""
    normalClass: str = "text-gray-500 bg-white hover:bg-gray-100 "
    "hover:text-gray-700 dark:bg-gray-800 dark:border-gray-700 "
    "dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white"


class Popover(Slottable, FlowbiteSvelteMixin, Component):
    title: str = ""
    defaultClass: str = "py-2 px-3"


# class Popper(Slottable, FlowbiteSvelteMixin, Component):

#     activeContent: bool = False
#     arrow: bool = True
#     offset: number = 8
#     placement: Placement = "top"
#     trigger: Literal["hover", "click"] = "hover"
#     triggeredBy: str = None
#     strategy: Literal["absolute", "fixed"] = "absolute"
#     open: bool = False
#     yOnly: bool = False


# class Frame(Slottable, FlowbiteSvelteMixin, Component):

#     tag: str = "div"
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
#     # transition: TransitionFunc = None
#     params: object = {}
#     # node: HTMLElement = None
#     # use: Action = noop
#     options: object = {}


class Progressbar(Slottable, FlowbiteSvelteMixin, Component):
    progress: number = 45
    size: Literal["h-2.5", "h-3", "h-4", "h-5"] = "h-2.5"
    labelInside: bool = False
    labelOutside: str = ""
    color: Literal[
        "blue", "gray", "red", "green", "yellow", "purple", "indigo"
    ] = "blue"
    labelInsideClass: str = (
        "text-blue-100 text-xs font-medium text-center p-0.5 leading-none "
        "rounded-full"
    )


class Rating(Slottable, FlowbiteSvelteMixin, Component):
    divClass: str = "flex items-center"
    size: str = "24"
    total: number = 5
    rating: number = 4
    ceil: bool = False
    count: bool = False


class AdvancedRating(Slottable, FlowbiteSvelteMixin, Component):
    divClass: str = "flex items-center mt-4"
    labelClass: str = "text-sm font-medium text-blue-600 dark:text-blue-500"
    ratingDivClass: str = "mx-4 w-2/4 h-5 bg-gray-200 rounded dark:bg-gray-700"
    ratingClass: str = "h-5 bg-yellow-400 rounded"
    rightLabelClass: str = "text-sm font-medium text-blue-600 dark:text-blue-500"
    unit: str = "%"


class ScoreRating(Slottable, FlowbiteSvelteMixin, Component):
    desc1Class: str = "bg-blue-100 w-8 text-blue-800 text-sm font-semibold "
    "inline-flex items-center p-1.5 rounded dark:bg-blue-200 dark:text-blue-800"
    desc2Class: str = "ml-2 w-24 font-medium text-gray-900 dark:text-white"
    desc3spanClass: str = "mx-2 w-1 h-1 bg-gray-900 rounded-full dark:bg-gray-500"
    desc3pClass: str = "text-sm w-24 font-medium text-gray-500 dark:text-gray-400"


class RatingComment(Slottable, FlowbiteSvelteMixin, Component):
    ceil: bool = False
    helpfullink: str = ""
    abuselink: str = ""


class Review(Slottable, FlowbiteSvelteMixin, Component):
    review: ReviewType = None
    articleClass: str = "md:gap-8 md:grid md:grid-cols-3"
    divClass: str = "flex items-center mb-6 space-x-4"
    imgClass: str = "w-10 h-10 rounded-full"
    ulClass: str = "space-y-4 text-sm text-gray-500 dark:text-gray-400"
    liClass: str = "flex items-center"


class Sidebar(Slottable, FlowbiteSvelteMixin, Component):
    asideClass: str = "w-64"


class SidebarBrand(Slottable, FlowbiteSvelteMixin, Component):
    site: SiteType = None


class SidebarCta(Slottable, FlowbiteSvelteMixin, Component):
    divWrapperClass: str = "p-4 mt-6 bg-blue-50 rounded-lg dark:bg-blue-900"
    divClass: str = "flex items-center mb-3"
    spanClass: str = "bg-orange-100 text-orange-800 text-sm font-semibold "
    "mr-2 px-2.5 py-0.5 rounded dark:bg-orange-200 dark:text-orange-900"
    label: str = ""


class SidebarDropdownItem(Slottable, FlowbiteSvelteMixin, Component):
    aClass: str = "flex items-center p-2 pl-11 w-full text-base font-normal "
    "text-gray-900 rounded-lg transition duration-75 group hover:bg-gray-100 "
    "dark:text-white dark:hover:bg-gray-700"
    href: str = ""
    label: str = ""
    activeClass: str = "flex items-center p-2 pl-11 text-base font-normal "
    "text-gray-900 bg-gray-200 dark:bg-gray-700 rounded-lg dark:text-white "
    "hover:bg-gray-100 dark:hover:bg-gray-700"
    active: bool = False

    # Events
    on_blur: Endpoint = None
    on_click: Endpoint = None
    on_focus: Endpoint = None
    on_keydown: Endpoint = None
    on_keypress: Endpoint = None
    on_keyup: Endpoint = None
    on_mouseenter: Endpoint = None
    on_mouseleave: Endpoint = None
    on_mouseover: Endpoint = None


class SidebarDropdownWrapper(Slottable, FlowbiteSvelteMixin, Component):
    btnClass: str = "flex items-center p-2 w-full text-base font-normal "
    "text-gray-900 rounded-lg transition duration-75 group hover:bg-gray-100 "
    "dark:text-white dark:hover:bg-gray-700"
    label: str = ""
    spanClass: str = "flex-1 ml-3 text-left whitespace-nowrap"
    ulClass: str = "py-2 space-y-2"
    isOpen: bool = False


class SidebarGroup(Slottable, FlowbiteSvelteMixin, Component):
    ulClass: str = "space-y-2"
    borderClass: str = "pt-4 mt-4 border-t border-gray-200 dark:border-gray-700"
    border: bool = False


class SidebarItem(Slottable, FlowbiteSvelteMixin, Component):
    aClass: str = "flex items-center p-2 text-base font-normal text-gray-900 "
    "rounded-lg dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700"
    href: str = ""
    label: str = ""
    spanClass: str = "ml-3"
    activeClass: str = "flex items-center p-2 text-base font-normal text-gray-900 "
    "bg-gray-200 dark:bg-gray-700 rounded-lg dark:text-white hover:bg-gray-100 "
    "dark:hover:bg-gray-700"
    active: bool = False

    # Events
    on_blur: Endpoint = None
    on_click: Endpoint = None
    on_focus: Endpoint = None
    on_keydown: Endpoint = None
    on_keypress: Endpoint = None
    on_keyup: Endpoint = None
    on_mouseenter: Endpoint = None
    on_mouseleave: Endpoint = None
    on_mouseover: Endpoint = None


class SidebarWrapper(Slottable, FlowbiteSvelteMixin, Component):
    divClass: str = "overflow-y-auto py-4 px-3 bg-gray-50 rounded dark:bg-gray-800"
    asideClass: str = "w-64"


class SpeedDial(Slottable, FlowbiteSvelteMixin, Component):
    defaultClass: str = "fixed right-6 bottom-6"
    placement: Placement = "top"
    pill: bool = True
    tooltip: Union[Placement, Literal["none"]] = "left"
    trigger: Literal["hover", "click"] = "hover"
    textOutside: bool = False
    # id: str = str(uuid.uuid4())

    # Events
    on_click: Endpoint = None


class SpeedDialButton(Slottable, FlowbiteSvelteMixin, Component):
    pass


class Spinner(Slottable, FlowbiteSvelteMixin, Component):
    color: Literal[
        "blue", "gray", "green", "red", "yellow", "pink", "purple", "white"
    ] = "blue"
    bg: str = "text-gray-300"
    size: str = "8"
    currentFill: str = "currentFill"
    currentColor: str = "currentColor"


class Tabs(Slottable, FlowbiteSvelteMixin, Component):
    style: Literal["full", "pill", "underline", "none"] = "none"
    defaultClass: str = "flex flex-wrap space-x-2"
    contentClass: str = "p-4 bg-gray-50 rounded-lg dark:bg-gray-800 mt-4"
    divider: bool = True
    activeClasses: str = (
        "p-4 text-blue-600 bg-gray-100 rounded-t-lg dark:bg-gray-800 "
        "dark:text-blue-500"
    )
    inactiveClasses: str = "p-4 text-gray-500 rounded-t-lg hover:text-gray-600 "
    "hover:bg-gray-50 dark:text-gray-400 dark:hover:bg-gray-800 "
    "dark:hover:text-gray-300"


class TabItem(Slottable, FlowbiteSvelteMixin, Component):
    open: bool = False
    title: str = "Tab title"
    activeClasses: str = ""
    inactiveClasses: str = ""
    defaultClass: str = (
        "inline-block text-sm font-medium text-center disabled:cursor-not-allowed"
    )


class Table(Slottable, FlowbiteSvelteMixin, Component):
    divClass: str = "relative overflow-x-auto"
    striped: bool = False
    hoverable: bool = False
    noborder: bool = False
    shadow: bool = False
    color: Literal[
        "blue",
        "green",
        "red",
        "yellow",
        "purple",
        "indigo",
        "pink",
        "default",
        "custom",
    ] = "default"


class TableBodyCell(Slottable, FlowbiteSvelteMixin, Component):
    tdClass: str = "px-6 py-4 whitespace-nowrap font-medium "


class TableBodyRow(Slottable, FlowbiteSvelteMixin, Component):
    # TODO: figure out how to get context
    color: Literal[
        "blue", "green", "red", "yellow", "purple", "default", "custom"
    ] = "purple"  # getContext("color")


class TableSearch(Slottable, FlowbiteSvelteMixin, Component):
    divClass: str = "relative overflow-x-auto shadow-md sm:rounded-lg"
    inputValue: str = ""
    striped: bool = False
    hoverable: bool = False
    placeholder: str = "Search"
    color: Literal[
        "blue", "green", "red", "yellow", "purple", "default", "custom"
    ] = "default"


class TableHead(Slottable, FlowbiteSvelteMixin, Component):
    theadClass: str = "text-xs uppercase"
    defaultRow: bool = True


class Timeline(Slottable, FlowbiteSvelteMixin, Component):
    customClass: str = ""
    order: Literal[
        "default", "vertical", "horizontal", "activity", "group", "custom"
    ] = "default"


class TimelineItem(Slottable, FlowbiteSvelteMixin, Component):
    title: str = ""
    date: str = ""
    customDiv: str = ""
    customTimeClass: str = ""


class Checkbox(Slottable, FlowbiteSvelteMixin, Component):
    color: FormColorType = "blue"
    title: str = ""
    date: str = ""


class TimelineHorizontal(Slottable, FlowbiteSvelteMixin, Component):
    olClass: str = "items-center sm:flex"


class TimelineItemHorizontal(Slottable, FlowbiteSvelteMixin, Component):
    title: str = ""
    date: str = ""


class Toast(Slottable, FlowbiteSvelteMixin, Component):
    color: Literal[
        "gray",
        "red",
        "yellow",
        "green",
        "indigo",
        "default",
        "purple",
        "pink",
        "blue",
        "light",
        "dark",
        "dropdown",
        "navbar",
        "navbarUl",
        "form",
        "none",
    ] = "blue"
    simple: bool = False
    position: Literal[
        "top-left", "top-right", "bottom-left", "bottom-right", "none"
    ] = "none"
    open: bool = True
    divClass: str = "w-full max-w-xs p-4"

    classes: str = None


class Tooltip(Slottable, FlowbiteSvelteMixin, Component):
    style: Literal["dark", "light", "auto", "custom"] = "dark"
    defaultClass: str = "py-2 px-3 text-sm font-medium"


class Activity(Slottable, FlowbiteSvelteMixin, Component):
    olClass: str = "relative border-l border-gray-200 dark:border-gray-700"


class ActivityItem(Slottable, FlowbiteSvelteMixin, Component):
    activities: List[ActivityType] = []


class Group(Slottable, FlowbiteSvelteMixin, Component):
    divClass: str = "p-5 mb-4 bg-gray-50 rounded-lg border border-gray-100 "
    "dark:bg-gray-800 dark:border-gray-700"
    timeClass: str = "text-lg font-semibold text-gray-900 dark:text-white"
    date: str = ""  # date: Union[Date, str] = ""


class GroupItem(Slottable, FlowbiteSvelteMixin, Component):
    timelines: List[GroupTimelineType] = []
    #     "blue", "red", "green", "yellow", "indigo", "purple", "pink"
    # ] = "blue"  # TODO: check that these are the right colors
    custom: bool = False
    inline: bool = False
    group: Union[number, str] = ""
    value: Union[number, str] = ""

    on_change: Endpoint = None
    on_click: Endpoint = None
    on_keydown: Endpoint = None
    on_keyup: Endpoint = None
    on_mouseenter: Endpoint = None
    on_mouseleave: Endpoint = None
    on_paste: Endpoint = None


class FloatingLabelInput(FlowbiteSvelteMixin, Component):
    # id: str = ""  #  determine if this should be uuid
    style: Literal["filled", "outlined", "standard"] = "standard"
    type: Literal["text"] = "text"  # TODO: add more input types
    size: Literal["small", "default"] = "default"
    color: Literal["base", "green", "red"] = "base"
    value: str = ""
    label: str = ""

    on_blur: Endpoint = None
    on_change: Endpoint = None
    on_click: Endpoint = None
    on_focus: Endpoint = None
    on_keydown: Endpoint = None
    on_keypress: Endpoint = None
    on_keyup: Endpoint = None
    on_mouseenter: Endpoint = None
    on_mouseleave: Endpoint = None
    on_mouseover: Endpoint = None
    on_paste: Endpoint = None


class Radio(Slottable, FlowbiteSvelteMixin, Component):
    color: Literal[
        "blue", "red", "green", "yellow", "indigo", "purple", "pink"
    ] = "blue"
    custom: bool = False
    inline: bool = False
    group: Union[number, str] = ""
    value: Union[number, str] = ""

    on_blur: Endpoint = None
    on_change: Endpoint = None
    on_click: Endpoint = None
    on_focus: Endpoint = None
    on_keydown: Endpoint = None
    on_keypress: Endpoint = None
    on_keyup: Endpoint = None
    on_mouseenter: Endpoint = None
    on_mouseleave: Endpoint = None
    on_mouseover: Endpoint = None
    on_paste: Endpoint = None


class Range(FlowbiteSvelteMixin, Component):
    value: number
    size: Literal["sm", "md", "lg"] = "md"
    min: number = None
    max: number = None
    step: number = None

    on_change: Endpoint = None
    on_click: Endpoint = None
    on_keydown: Endpoint = None
    on_keypress: Endpoint = None
    on_keyup: Endpoint = None


class Search(FlowbiteSvelteMixin, Component):
    size: Literal["sm", "md", "lg"] = "lg"
    placeholder: str = "Search"
    value: Union[str, number] = ""

    on_blur: Endpoint = None
    on_change: Endpoint = None
    on_click: Endpoint = None
    on_focus: Endpoint = None
    on_keydown: Endpoint = None
    on_keypress: Endpoint = None
    on_keyup: Endpoint = None
    on_mouseenter: Endpoint = None
    on_mouseleave: Endpoint = None
    on_mouseover: Endpoint = None
    on_paste: Endpoint = None


class Select(Slottable, FlowbiteSvelteMixin, Component):
    items: list
    value: Union[str, number]
    placeholder: str = "Choose option ..."
    underline: bool = False
    size: Literal["sm", "md", "lg"] = "md"

    on_change: Endpoint = None
    on_input: Endpoint = None


class Textarea(FlowbiteSvelteMixin, Component):
    value: str = ""
    rows: int = 3
    placeholder: str = ""

    classes: str = None

    on_blur: Endpoint = None
    on_change: Endpoint = None
    on_click: Endpoint = None
    on_focus: Endpoint = None
    on_input: Endpoint = None
    on_keydown: Endpoint = None
    on_keypress: Endpoint = None
    on_keyup: Endpoint = None
    on_mouseenter: Endpoint = None
    on_mouseleave: Endpoint = None
    on_mouseover: Endpoint = None
    on_paste: Endpoint = None


class Toggle(Slottable, FlowbiteSvelteMixin, Component):
    size: Literal["small", "default", "large"] = "default"
    group: List[Union[str, number]] = ()
    value: Union[str, number] = ""
    checked: bool = False

    on_change: Endpoint = None
    on_click: Endpoint = None


"""Typography."""


class A(Slottable, FlowbiteSvelteMixin, Component):
    href: str = "#"
    color: str = "text-blue-600 dark:text-blue-500"
    aClass: str = "inline-flex items-center hover:underline"


class Blockquote(Slottable, FlowbiteSvelteMixin, Component):
    border: bool = False
    italic: bool = True
    borderClass: str = "border-l-4 border-gray-300 dark:border-gray-500"
    bgClass: str = "bg-gray-50 dark:bg-gray-800"
    bg: bool = False
    baseClass: str = "font-semibold text-gray-900 dark:text-white"
    alignment: Literal["left", "center", "right"] = "left"
    size: Literal[
        "xs",
        "sm",
        "base",
        "lg",
        "xl",
        "2xl",
        "3xl",
        "4xl",
        "5xl",
        "6xl",
        "7xl",
        "8xl",
        "9xl",
    ] = "lg"


class DescriptionList(Slottable, FlowbiteSvelteMixin, Component):
    tag: Literal["dt", "dd"] = ""
    dtClass: str = "text-gray-500 md:text-lg dark:text-gray-400"
    ddClass: str = "text-lg font-semibold"


class Heading(Slottable, FlowbiteSvelteMixin, Component):
    tag: Literal["h1", "h2", "h3", "h4", "h5", "h6"] = "h1"
    color: str = "text-gray-900 dark:text-white"
    customSize: str = ""


class Hr(Slottable, FlowbiteSvelteMixin, Component):
    icon: bool = False
    width: str = "w-full"
    height: str = "h-px"
    divClass: str = "inline-flex justify-center items-center w-full"
    hrClass: str = "bg-gray-200 rounded border-0 dark:bg-gray-700"
    iconDivClass: str = "absolute left-1/2 px-4 bg-white -translate-x-1/2 "
    textSpanClass: str = "absolute left-1/2 px-3 font-medium text-gray-900 "
    "bg-white -translate-x-1/2 dark:text-white "
    middleBgColor: str = "dark:bg-gray-900"


class Layout(Slottable, FlowbiteSvelteMixin, Component):
    divClass: str = "grid"
    cols: str = "grid-cols-1 sm:grid-cols-2"
    gap: number = 4


class Li(Slottable, FlowbiteSvelteMixin, Component):
    icon: bool = False
    liClass: str = ""


class List(Slottable, FlowbiteSvelteMixin, Component):
    tag: Literal["ul", "ol", "dl"] = "ul"
    list: Literal["disc", "none", "decimal"] = "disc"
    position: Literal["inside", "outside"] = "inside"
    color: str = "text-gray-500 dark:text-gray-400"
    olClass: str = "list-decimal list-inside"
    ulClass: str = "max-w-md"
    dlClass: str = "max-w-md divide-y divide-gray-200 dark:divide-gray-700"


class Mark(Slottable, FlowbiteSvelteMixin, Component):
    color: str = "text-white dark:bg-blue-500"
    bgColor: str = "bg-blue-600"
    markClass: str = "px-2 rounded"


class P(Slottable, FlowbiteSvelteMixin, Component):
    color: str = "text-gray-900 dark:text-white"
    height: Literal["normal", "relaxed", "loose"] = "normal"
    align: Literal["left", "center", "right"] = "left"
    justify: bool = False
    italic: bool = False
    firstupper: bool = False
    upperClass: str = "first-line:uppercase first-line:tracking-widest "
    "first-letter:text-7xl first-letter:font-bold first-letter:text-gray-900 "
    "dark:first-letter:text-gray-100 first-letter:mr-3 first-letter:float-left"
    opacity: Union[number, None] = None
    whitespace: Literal["normal", "nowrap", "pre", "preline", "prewrap"] = "normal"
    size: Literal[
        "xs",
        "sm",
        "base",
        "lg",
        "xl",
        "2xl",
        "3xl",
        "4xl",
        "5xl",
        "6xl",
        "7xl",
        "8xl",
        "9xl",
    ] = "base"
    space: Literal[
        "tighter",
        "tight",
        "normal",
        "wide",
        "wider",
        "widest",
        None,
    ] = None
    weight: Literal[
        "thin",
        "extralight",
        "light",
        "normal",
        "medium",
        "semibold",
        "bold",
        "extrabold",
        "black",
    ] = "normal"


class Secondary(Slottable, FlowbiteSvelteMixin, Component):
    color: str = "text-gray-500 dark:text-gray-400"
    secondaryClass: str = "font-semibold"


class Span(Slottable, FlowbiteSvelteMixin, Component):
    italic: bool = False
    underline: bool = False
    linethrough: bool = False
    uppercase: bool = False
    gradient: bool = False
    highlight: bool = False
    highlightClass: str = "text-blue-600 dark:text-blue-500"
    decorationClass: str = "decoration-2 decoration-blue-400 "
    "dark:decoration-blue-600"
    gradientClass: str = (
        "text-transparent bg-clip-text bg-gradient-to-r to-emerald-600 "
        "from-sky-400"
    )


__all__ = [
    "Accordion",
    "AccordionItem",
    "Button",
    "Card",
    "Dropdown",
    "DropdownItem",
    "DropdownDivider",
    "DropdownHeader",
]
