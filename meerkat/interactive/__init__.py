import pathlib

from .app.src.lib.component import contrib, core, flowbite, html, plotly
from .app.src.lib.component.abstract import BaseComponent, Component
from .app.src.lib.component.contrib.fm_filter import FMFilter
from .app.src.lib.component.contrib.mocha import ChangeList
from meerkat.interactive.app.src.lib.component.core import *
from .app.src.lib.shared.cell.website import Website
from .edit import EditTarget
from .endpoint import Endpoint, endpoint, endpoints, make_endpoint
from .graph import (  # noqa: F401
    Store,
    is_unmarked_context,
    magic,
    mark,
    reactive,
    unmarked,
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
    # Component
    "BaseComponent",
    "Component",
    # Page
    "Page",
    "page",
    # Store
    "Store",
    # Endpoint
    "Endpoint",
    "endpoint",
    "endpoints",
    "make_endpoint",
    # Reactivity
    "reactive",
    "unmarked",
    "is_unmarked_context",
    "mark",
    "magic",
    # Modification Types
    "DataFrameModification",
    "Modification",
    # Add-ons
    "State",
    # <<<< Component Namespaces >>>>
    "contrib",
    "core",
    "flowbite",
    "html",
    "plotly",
    # <<<< Contrib Components >>>>
    "ChangeList",
    "FMFilter",
    # <<<< Shared Components >>>>
    "Website",
    # <<<< Utilities >>>>
    "print",
]

# Add core components to the top-level namespace.
__all__.extend(core.__all__)
