import pathlib

from .app.src.lib.component import contrib, core, flowbite, html, plotly
from .app.src.lib.component.abstract import BaseComponent, Component
from .app.src.lib.component.core.button import Button
from .app.src.lib.component.core.choice import Choice
from .app.src.lib.component.core.codedisplay import CodeDisplay
from .app.src.lib.component.core.document import Document
from .app.src.lib.component.core.editor import Editor
from .app.src.lib.component.core.filter import Filter
from .app.src.lib.component.core.gallery import Gallery
from .app.src.lib.component.core.image import Image
from .app.src.lib.component.core.markdown import Markdown
from .app.src.lib.component.core.match import Match
from .app.src.lib.component.core.multiselect import MultiSelect
from .app.src.lib.component.contrib.mocha import ChangeList
from .app.src.lib.component.core.put import Put
from .app.src.lib.component.core.slicebycards import SliceByCards
from .app.src.lib.component.core.sort import Sort
from .app.src.lib.component.core.stats import Stats
from .app.src.lib.component.core.table import Table
from .app.src.lib.component.core.tabs import Tabs
from .app.src.lib.component.core.text import Text
from .app.src.lib.component.core.textbox import Textbox
from .app.src.lib.component.core.toggle import Toggle
from .app.src.lib.shared.cell.code import Code
from .edit import EditTarget
from .endpoint import Endpoint, endpoint, endpoints, make_endpoint
from .graph import (
    Store,
    StoreFrontend,
    is_reactive,
    make_store,
    no_react,
    react,
    reactive,
    trigger,
)
from .modification import DataFrameModification, Modification
from .page import Page, page
from .startup import start
from .state import State
from .utils import print

INTERACTIVE_LIB_PATH = pathlib.Path(__file__).parent.resolve()

__all__ = [
    # <<<< Startup >>>>
    "start",
    # <<<< Core Library >>>>
    ## Component
    "BaseComponent",
    "Component",
    ## Page
    "Page",
    "page",
    ## Store
    "Store",
    "StoreFrontend",
    "make_store",
    ## Endpoint
    "Endpoint",
    "endpoint",
    "endpoints",
    "make_endpoint",
    ## Reactivity
    "is_reactive",
    "no_react",
    "react",
    "reactive",
    "trigger",
    ## Modification Types
    "DataFrameModification",
    "Modification",
    ## Add-ons
    "State",
    # <<<< Component Namespaces >>>>
    "contrib",
    "core",
    "flowbite",
    "html",
    "plotly",
    # <<<< Core Components >>>>
    "Button",
    "Choice",
    "Code",
    "CodeDisplay",
    "Document",
    "EditTarget",
    "Editor",
    "Filter",
    "Gallery",
    "Image",
    "Markdown",
    "Match",
    "MultiSelect",
    "SliceByCards",
    "Sort",
    "Stats",
    "Table",
    "Tabs",
    "Text",
    "Textbox",
    "Toggle",
    "Put",
    # <<<< Utilities >>>>
    "print",
]
