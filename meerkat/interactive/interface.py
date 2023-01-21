import code
import os
from functools import partial, wraps
from typing import Callable

import rich
from IPython.display import IFrame
from pydantic import BaseModel

from meerkat.interactive.app.src.lib.component.abstract import (
    Component,
    ComponentFrontend,
)
from meerkat.interactive.svelte import SvelteWriter
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.state import state


def interface(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        interface = Interface(component=partial(fn, *args, **kwargs))
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
        id: str,
        name: str = "Interface",
        height: str = "1000px",
        width: str = "100%",
    ):

        super().__init__(id=id)

        self.component = component
        self.name = name
        self.height = height
        self.width = width

        # Call `init_run`
        svelte_writer = SvelteWriter()
        svelte_writer.init_run()

    def __call__(self):
        """Return the FastAPI object, this allows Interface objects to be
        targeted by uvicorn when running a script."""
        from meerkat.interactive.api import MeerkatAPI

        return MeerkatAPI

    def launch(self, return_url: bool = False):
        from meerkat.interactive.startup import is_notebook, output_startup_message

        if state.frontend_info is None:
            rich.print(
                "Frontend is not initialized. Running `mk.gui.start()`."
            )
            from .startup import start
            start()

        url = f"{state.frontend_info.url}/{self.id}"

        if return_url:
            return url

        if is_notebook():
            return IFrame(url, width=self.width, height=self.height)
        else:

            rich.print(
                ":scroll: "
                f"Interface [violet]{self.id}[/violet] "
                f"is at [violet]{url}[/violet]"
            )
            rich.print(
                ":newspaper: "
                f"API docs are at [violet]{state.api_info.docs_url}[/violet]"
            )
            rich.print()

            in_mk_run_subprocess = int(os.environ.get("MEERKAT_RUN", 0))
            if not in_mk_run_subprocess:
                # get locals of the main module when running in script.
                import __main__

                code.interact(local=__main__.__dict__)

    @property
    def frontend(self):
        return InterfaceFrontend(name=self.name, component=self.component.frontend)
