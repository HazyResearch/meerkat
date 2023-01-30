import code
import os
from functools import partial, wraps
from typing import Callable

import rich
from IPython.display import IFrame
from pydantic import BaseModel

from meerkat.interactive import html
from meerkat.interactive.app.src.lib.component._internal.progress import Progress
from meerkat.interactive.app.src.lib.component.abstract import (
    BaseComponent,
    ComponentFrontend,
)
from meerkat.interactive.svelte import SvelteWriter
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.state import state


def page(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        page = Page(
            component=partial(fn, *args, **kwargs),
            id=fn.__name__,
        )
        return page.launch()

    return wrapper


class PageFrontend(BaseModel):
    component: ComponentFrontend
    name: str


class Page(IdentifiableMixin):

    _self_identifiable_group: str = "pages"

    def __init__(
        self,
        component: BaseComponent,
        id: str,
        name: str = "Page",
        height: str = "1000px",
        width: str = "100%",
    ):

        super().__init__(id=id)

        self.component = html.div(
            slots=[
                Progress(),
                component,
            ],
        )
        self.name = name
        self.height = height
        self.width = width

        # Call `init_run`
        # KG: TODO: figure out if we need this here.
        svelte_writer = SvelteWriter()
        svelte_writer.init_run()

    def __call__(self):
        """Return the FastAPI object, this allows Page objects to be
        targeted by uvicorn when running a script."""
        from meerkat.interactive.api import MeerkatAPI

        return MeerkatAPI

    def launch(self, return_url: bool = False):
        from meerkat.interactive.startup import is_notebook

        if state.frontend_info is None:
            rich.print("Frontend is not initialized. Running `mk.gui.start()`.")
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
                f"Frontend [violet]{self.id}[/violet] "
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
        return PageFrontend(name=self.name, component=self.component.frontend)
