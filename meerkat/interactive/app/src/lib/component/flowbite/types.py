from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Union

number = Union[int, float]
AlignType = Literal["text-center", "text-left", "text-right"]
ButtonType = Literal["button", "submit", "reset"]
Buttontypes = Literal[
    "blue",
    "blue-outline",
    "dark",
    "dark-outline",
    "light",
    "green",
    "green-outline",
    "red",
    "red-outline",
    "yellow",
    "yellow-outline",
    "purple",
    "purple-outline",
]
Buttonshadows = Literal[
    "blue", "green", "cyan", "teal", "lime", "red", "pink", "purple"
]

Colors = Literal[
    "blue",
    "gray",
    "red",
    "yellow",
    "purple",
    "green",
    "indigo",
    "pink",
    "white",
    "custom",
]
SizeType = Literal["sm", "md", "lg"]

Placement = Literal["top", "bottom", "left", "right"]


@dataclass
class ActivityType:
    title: str  # title: Union[HTMLElement, str]
    date: str  # date: Union[Date, str]
    src: str
    alt: str
    text: str = None  # text: Optional[Union[HTMLElement, str]] = None


@dataclass
class AuthFieldType:
    label: str
    fieldtype: str
    required: Optional[bool] = None
    placeholder: Optional[str] = None


@dataclass
class CheckboxType:
    id: str
    label: str
    checked: Optional[bool] = None
    disabled: Optional[bool] = None
    helper: Optional[str] = None


Colors = Literal[
    "blue",
    "gray",
    "red",
    "yellow",
    "purple",
    "green",
    "indigo",
    "pink",
    "white",
    "custom",
]


@dataclass
class DotType:
    top: Optional[bool] = None
    color: Optional[str] = None


DrawerTransitionTypes = Optional[
    Literal[
        "fade",
        "fly",
        "slide",
        "blur",
        "in:fly",
        "out:fly",
        "in:slide",
        "out:slide",
        "in:fade",
        "out:fade",
        "in:blur",
        "out:blur",
    ]
]


FormColorType = Literal["blue", "red", "green", "purple", "teal", "yellow", "orange"]
Gradientduotones = Literal[
    "purple2blue",
    "cyan2blue",
    "green2blue",
    "purple2pink",
    "pink2orange",
    "teal2lime",
    "red2yellow",
]


@dataclass
class IconType:
    name: str
    size: Optional[int] = None
    color: Optional[Colors] = None
    class_: Optional[str] = None


@dataclass
class ImgType:
    src: str
    alt: Optional[str] = None


InputType = Literal[
    "color",
    "date",
    "datetime-local",
    "email",
    "file",
    "hidden",
    "image",
    "month",
    "number",
    "password",
    "reset",
    "submit",
    "tel",
    "text",
    "time",
    "url",
    "week",
    "search",
]


@dataclass
class InteractiveTabType:
    name: str
    id: int
    content: str
    active: Optional[bool] = None
    disabled: Optional[bool] = None
    icon: Optional[IconType] = None
    iconSize: Optional[int] = None


@dataclass
class ListGroupItemType:
    current: Optional[bool] = None
    disabled: Optional[bool] = None
    href: Optional[str] = None


@dataclass
class LinkType:
    name: str
    href: Optional[str] = None
    rel: Optional[str] = None
    active: Optional[bool] = None


@dataclass
class ListCardType:
    img: ImgType
    field1: str
    field2: str = ""
    field3: str = ""


@dataclass
class NavbarType:
    name: str
    href: str
    rel: str = ""
    child: List["NavbarType"] = None


@dataclass
class PageType:
    pageNum: number
    href: str


SizeType = Literal["xs", "sm", "md", "lg", "xl"]
FormSizeType = Literal["sm", "md", "lg"]


@dataclass
class PillTabType:
    name: str
    selected: bool
    href: str


@dataclass
class ReviewType:
    name: str
    imgSrc: str
    imgAlt: str
    title: str
    rating: number
    address: Optional[str] = None
    reviewDate: Optional[str] = None
    item1: Optional[str] = None
    item2: Optional[str] = None
    item3: Optional[str] = None


@dataclass
class SelectOptionType:
    name: Union[str, number]
    value: Union[str, number]


@dataclass
class SidebarType:
    id: number
    name: str
    href: Optional[str] = None
    # icon: Optional[Type[SvelteComponent]] = None
    iconSize: Optional[number] = None
    iconClass: Optional[str] = None
    iconColor: Optional[str] = None
    rel: Optional[str] = None
    # children: Optional[List[SidebarType]] = None
    # subtext: Optional[HTMLElement] = None


@dataclass
class SidebarCtaType:
    label: str
    # text: HTMLElement


@dataclass
class SiteType:
    name: str
    href: str
    img: Optional[str] = None


@dataclass
class SocialMediaLinkType:
    parent: str
    children: Optional[List[LinkType]] = None


@dataclass
class SocialMediaType:
    href: str
    # icon: Type[SvelteComponent]
    iconSize: Optional[number] = None
    iconClass: Optional[str] = None


@dataclass
class TabHeadType:
    name: str
    id: number


@dataclass
class TabType:
    name: str
    active: bool
    href: str
    rel: Optional[str] = None


@dataclass
class TableDataHelperType:
    start: number
    end: number
    total: number


@dataclass
class TimelineItemType:
    date: str  # date: Union[Date, str]
    title: str
    # icon: Optional[Type[SvelteComponent]]
    href: Optional[str]
    linkname: Optional[str]
    text: str  # text: Optional[Union[HTMLElement, str]]


@dataclass
class TimelineItemVerticalType:
    date: str  # date: Union[Date, str]
    title: str
    # icon: Optional[Type[SvelteComponent]]
    iconSize: Optional[int]
    iconClass: Optional[str]
    href: Optional[str]
    linkname: Optional[str]
    text: str  # text: Optional[Union[HTMLElement, str]]


@dataclass
class TimelineItemHorizontalType:
    date: str  # date: Union[Date, str]
    title: str
    # icon: Optional[Type[SvelteComponent]]
    iconSize: Optional[int]
    iconClass: Optional[str]
    text: str  # text: Optional[Union[HTMLElement, str]]


@dataclass
class TransitionParamTypes:
    delay: Optional[int]
    duration: Optional[int]
    easing: Optional[Callable[[int], int]]
    css: Optional[Callable[[int, int], str]]
    tick: Optional[Callable[[int, int], None]]


Textsize = Literal[
    "text-xs",
    "text-sm",
    "text-base",
    "text-lg",
    "text-xl",
    "text-2xl",
    "text-3xl",
    "text-4xl",
]
ToggleColorType = Literal["blue", "red", "green", "purple", "yellow", "teal", "orange"]
TransitionTypes = Literal[
    "fade",
    "fly",
    "slide",
    "blur",
    "in:fly",
    "out:fly",
    "in:slide",
    "out:slide",
    "in:fade",
    "out:fade",
    "in:blur",
    "out:blur",
]

Colors = Literal[
    "blue",
    "gray",
    "red",
    "yellow",
    "purple",
    "green",
    "indigo",
    "pink",
    "white",
    "custom",
]


@dataclass
class drawerTransitionParamTypes:
    amount: Optional[int] = None
    delay: Optional[int] = None
    duration: Optional[int] = None
    easing: Callable = None
    opacity: Optional[number] = None
    x: Optional[int] = None
    y: Optional[int] = None


drawerTransitionTypes = Optional[
    Literal[
        "fade",
        "fly",
        "slide",
        "blur",
        "in:fly",
        "out:fly",
        "in:slide",
        "out:slide",
        "in:fade",
        "out:fade",
        "in:blur",
        "out:blur",
    ]
]


@dataclass
class DropdownType:
    name: str
    href: str
    divider: Optional[bool] = None


FormColorType = Literal["blue", "red", "green", "purple", "teal", "yellow", "orange"]
Gradientduotones = Literal[
    "purple2blue",
    "cyan2blue",
    "green2blue",
    "purple2pink",
    "pink2orange",
    "teal2lime",
    "red2yellow",
]


@dataclass
class GroupTimelineType:
    title: str
    src: str
    alt: str
    href: Optional[str] = None
    isPrivate: Optional[bool] = None
    comment: Optional[str] = None


@dataclass
class IconType:
    name: str
    size: Optional[int] = None
    color: Optional[Colors] = None
    class_: Optional[str] = None


@dataclass
class IconTabType:
    name: str
    active: bool
    href: str
    rel: Optional[str] = None
    icon: Optional[str] = None
    iconSize: Optional[int] = None


@dataclass
class ImgType:
    src: str
    alt: Optional[str] = None


InputType = Literal[
    "color",
    "date",
    "datetime-local",
    "email",
    "file",
    "hidden",
    "image",
    "month",
    "number",
    "password",
    "reset",
    "submit",
    "tel",
    "text",
    "time",
    "url",
    "week",
    "search",
]


SizeType = Literal["xs", "sm", "md", "lg", "xl"]
FormSizeType = Literal["sm", "md", "lg"]
