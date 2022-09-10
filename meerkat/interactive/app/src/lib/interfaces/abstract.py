import code
import sys
from functools import partial, wraps
from typing import Callable, List

from IPython.display import IFrame
from pydantic import BaseModel

from meerkat.interactive.app.src.lib.component.abstract import (
    Component,
    ComponentConfig,
)
from meerkat.interactive.graph import Pivot, PivotConfig
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.state import state


class InterfaceConfig(BaseModel):

    pivots: List[PivotConfig]
    components: List[ComponentConfig]
    name: str = "Interface"


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


class InterfaceMeta(type):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        # Automatically call the layout method when the
        # interface is created
        instance._layout()
        return instance


class Interface(IdentifiableMixin, metaclass=InterfaceMeta):
    # TODO (all): I think this should probably be a subclassable thing that people
    # implement. e.g. TableInterface

    identifiable_group: str = "interfaces"

    def __init__(self, layout: callable = None):
        if layout is not None:
            self.layout = layout
        super().__init__()

        self.pivots = []
        self.components = []

    def _layout(self):
        frame, _ = call_function_get_frame(self.layout)
        if len(self.components) == 0:
            # Inspect the local frame of the layout function
            # and add all the components to self.components
            # in the order in which they were defined
            for _, val in frame.f_locals.items():
                if isinstance(val, Component):
                    self.components.append(val)

    def layout(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def launch(self, return_url: bool = False):
        from meerkat.interactive.startup import is_notebook, output_startup_message

        if state.network_info is None:
            raise ValueError(
                "Interactive mode not initialized."
                "Run `network, register_api = mk.interactive_mode()` followed by "
                "`register_api()` first."
            )

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

    def pivot(self, obj):
        # checks whether the object is valid pivot

        pivot = Pivot(obj)
        self.pivots.append(pivot)

        return pivot

    @property
    def config(self):
        return InterfaceConfig(
            pivots=[pivot.config for pivot in self.pivots],
            components=[component.config for component in self.components],
        )
