import code
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import Callable, Dict, List, Union

from fastapi import HTTPException
from IPython.display import IFrame
from pydantic import BaseModel
from meerkat.interactive.app.src.lib.component.abstract import Component, ComponentFrontend
from meerkat.interactive.frontend import FrontendMixin

# from meerkat.interactive.app.src.lib.component.abstract import (
#     Component,
#     ComponentSchema,
# )
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.state import state
from meerkat.tools.utils import nested_apply


def interface(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        interface = Interface(layout=partial(fn, *args, **kwargs))
        return interface.launch()

    return wrapper


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
    components: Union[List[ComponentFrontend], Dict[str, ComponentFrontend]]
    name: str


class Interface(IdentifiableMixin):
    # TODO (all): I think this should probably be a subclassable thing that people
    # implement. e.g. TableInterface

    _self_identifiable_group: str = "interfaces"

    def __init__(
        self,
        components: Union[List[Component], Dict[str, Component]] = None,
        layout: Layout = None,
        name: str = "Interface",
        id: str = None,
    ):

        super().__init__(id=id)

        self.name = name

        self.layout = layout
        if self.layout is None:
            self.layout = Layout()

        self.components = components
        if self.components is None:
            self.components = []

    def get(self, id: str):
        try:
            from meerkat.state import state

            interface = state.identifiables.get(id, "interfaces")
        except KeyError:
            raise HTTPException(
                status_code=404, detail="No interface with id {}".format(id)
            )
        return interface

    def launch(self, return_url: bool = False):
        from meerkat.interactive.startup import is_notebook, output_startup_message

        if state.network_info is None:
            raise ValueError(
                "Interactive mode not initialized."
                "Run `network = mk.gui.start()` first."
            )

        if state.network_info.shareable_npm_server_name is not None:
            url = (
                f"{state.network_info.shareable_npm_server_url}/interface?id={self.id}"
            )
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
            components=nested_apply(self.components, lambda c: c.frontend if isinstance(c, FrontendMixin) else c),
        )
