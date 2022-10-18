import code
import sys
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import Callable, Dict, List, Union

from IPython.display import IFrame
from pydantic import BaseModel

from meerkat.interactive.app.src.lib.component.abstract import (
    Component,
    ComponentConfig,
)
from meerkat.interactive.graph import Pivot, PivotConfig
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.state import state
from meerkat.tools.utils import nested_apply


def call_function_get_frame(func, *args, **kwargs):
    """https://stackoverflow.com/questions/4214936/how-can-i-get-the-values-of-
    the-locals-of-a-function-after-it-has-been-executed Calls the function
    *func* with the specified arguments and keyword arguments and snatches its
    local frame before it actually executes."""

    frame = None
    trace = sys.gettrace()

    def snatch_locals(_frame, name, arg):
        nonlocal frame
        if frame is None and name == "call":
            frame = _frame
            sys.settrace(trace)
        return trace

    sys.settrace(snatch_locals)
    try:
        result = func(*args, **kwargs)
    finally:
        sys.settrace(trace)
    return frame, result


def interface(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        interface = Interface(layout=partial(fn, *args, **kwargs))
        return interface.launch()

    return wrapper
    # frame, out = call_function_get_frame(self.build)

    # if out is not None:

    # if components is not None:
    #     self.components = components

    # if len(self.components) == 0:
    #     # Inspect the local frame of the build function
    #     # and add all the components to self.components
    #     # in the order in which they were defined
    #     for _, val in frame.f_locals.items():
    #         if isinstance(val, Component):
    #             self.components.append(val)


class LayoutConfig(BaseModel):
    name: str
    props: Dict


@dataclass
class Layout:
    name: str = "DefaultLayout"
    props: Dict[str, any] = field(default_factory=dict)

    @property
    def config(self):
        return LayoutConfig(name=self.name, props=self.props)


class InterfaceConfig(BaseModel):

    layout: LayoutConfig
    components: Union[List[ComponentConfig], Dict[str, ComponentConfig]]
    name: str


class Interface(IdentifiableMixin):
    # TODO (all): I think this should probably be a subclassable thing that people
    # implement. e.g. TableInterface

    identifiable_group: str = "interfaces"

    def __init__(
        self,
        components: Union[List[Component], Dict[str, Component]] = None,
        layout: Layout = None,
        name: str = "Interface",
    ):

        super().__init__()

        self.name = name

        self.layout = layout
        if self.layout is None:
            self.layout = Layout()

        self.components = components
        if self.components is None:
            self.components = []

    def launch(self, return_url: bool = False):
        from meerkat.interactive.startup import is_notebook, output_startup_message

        if state.network_info is None:
            raise ValueError(
                "Interactive mode not initialized."
                "Run `network = mk.gui.start()` first."
            )

        if state.network_info.shareable_npm_server_name is not None:
            url = f"{state.network_info.shareable_npm_server_url}/interface?id={self.id}"
        else:
            url = f"{state.network_info.npm_server_url}/interface?id={self.id}"
        
        if return_url:
            return url
        if is_notebook():
            return IFrame(url, width="100%", height="1000")
        else:
            import webbrowser

            webbrowser.open(url)

            output_startup_message(url=url)

            # get locals of the main module when running in script.
            import __main__

            code.interact(local=__main__.__dict__)

    @property
    def config(self):
        return InterfaceConfig(
            name=self.name,
            layout=self.layout.config,
            components=nested_apply(self.components, lambda c: c.config),
        )
