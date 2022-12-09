import code
from functools import partial, wraps
from typing import Callable

from fastapi import HTTPException
from IPython.display import IFrame
from pydantic import BaseModel

from meerkat.interactive.app.src.lib.component.abstract import (
    Component,
    ComponentFrontend,
)
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.state import state


def interface(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        interface = Interface(layout=partial(fn, *args, **kwargs))
        return interface.launch()

    return wrapper


class InterfaceFrontend(BaseModel):
    component: ComponentFrontend
    name: str


class Interface(IdentifiableMixin):

    _self_identifiable_group: str = "interfaces"

    def __init__(
        self,
        component: Component,
        name: str = "Interface",
        id: str = None,
        height: str = "1000px",
        width: str = "100%",
    ):

        super().__init__(id=id)

        self.component = component
        self.name = name
        self.height = height
        self.width = width

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
            url = f"{state.network_info.shareable_npm_server_url}?id={self.id}"
        else:
            url = f"{state.network_info.npm_server_url}?id={self.id}"

        if return_url:
            return url
        if is_notebook():
            return IFrame(url, width=self.width, height=self.height)
        else:
            import webbrowser

            webbrowser.open(url)

            output_startup_message(url=url)

            # get locals of the main module when running in script.
            import __main__

            code.interact(local=__main__.__dict__)

    @property
    def frontend(self):
        return InterfaceFrontend(name=self.name, component=self.component.frontend)
