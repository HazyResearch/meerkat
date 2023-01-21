import pathlib

from meerkat.interactive.app.src.lib.component import flowbite, html
from meerkat.interactive.app.src.lib.component.codedisplay import CodeDisplay
from meerkat.interactive.app.src.lib.component.multiselect import MultiSelect
from meerkat.interactive.app.src.lib.layouts import (
    ColumnLayout,
    Div,
    Flex,
    Grid,
    RowLayout,
)

# from meerkat.interactive.app.src.lib.shared.cell.basic import Text
from meerkat.interactive.app.src.lib.shared.cell.code import Code
from meerkat.interactive.app.src.lib.shared.cell.image import Image
from meerkat.interactive.graph import (
    Store,
    StoreFrontend,
    is_reactive,
    make_store,
    no_react,
    react,
    reactive,
    trigger,
)

from .app.src.lib.component.abstract import AutoComponent, Component
from .app.src.lib.component.button import Button
from .app.src.lib.component.choice import Choice
from .app.src.lib.component.discover import Discover
from .app.src.lib.component.document import Document
from .app.src.lib.component.editor import Editor
from .app.src.lib.component.filter import Filter
from .app.src.lib.component.gallery import Gallery
from .app.src.lib.component.markdown import Markdown
from .app.src.lib.component.match import Match
from .app.src.lib.component.plot import Plot
from .app.src.lib.component.row import Row
from .app.src.lib.component.slicebycards import SliceByCards
from .app.src.lib.component.sort import Sort
from .app.src.lib.component.stats import Stats
from .app.src.lib.component.stats_labeler import StatsLabeler
from .app.src.lib.component.table import Table
from .app.src.lib.component.tabs import Tabs
from .app.src.lib.component.textbox import Textbox
from .app.src.lib.component.toggle import Toggle
from .edit import EditTarget
from .endpoint import Endpoint, endpoint, endpoints, make_endpoint
from .interface import Interface, interface
from .modification import DataFrameModification, Modification
from .startup import start
from .state import State

INTERACTIVE_LIB_PATH = pathlib.Path(__file__).parent.resolve()

__all__ = [
    "flowbite",
    "Endpoint",
    "State",
    "endpoint",
    "endpoints",
    "react",
    "no_react",
    "is_reactive",
    "make_endpoint",
    "Modification",
    "DataFrameModification",
    "Document",
    "Store",
    "StoreFrontend",
    "make_store",
    "trigger",
    "AutoComponent",
    "Component",
    "Discover",
    "Gallery",
    "Markdown",
    "Match",
    "Row",
    "Plot",
    "SliceByCards",
    "Stats",
    "EditTarget",
    "Table",
    "Tabs",
    "Filter",
    "Interface",
    "interface",
    "reactive",
    "start",
    "Editor",
    "Sort",
    "StatsLabeler",
    "Choice",
    "Textbox",
    "Button",
    "ColumnLayout",
    "RowLayout",
    "Div",
    "Flex",
    "Grid",
    "MultiSelect",
    "Image",
    "Text",
    "Code",
    "CodeDisplay",
    "Toggle",
]
