import code
from functools import partial, wraps
from typing import Callable

import rich
from pydantic import BaseModel

from meerkat.constants import MEERKAT_RUN_SUBPROCESS, is_notebook
from meerkat.interactive import html
from meerkat.interactive.app.src.lib.component._internal.progress import Progress
from meerkat.interactive.app.src.lib.component.abstract import (
    BaseComponent,
    ComponentFrontend,
)
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.state import state

if is_notebook():
    from IPython.display import IFrame


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
        height: str = "100%",
        width: str = "100%",
        progress: bool = False,
    ):

        super().__init__(id=id)

        if progress:
            component = html.flexcol(
                slots=[
                    Progress(),
                    component,
                ],
                classes="h-full",
            )
        self.component = component
        self.name = name
        self.height = height
        self.width = width

    def __call__(self):
        """Return the FastAPI object, this allows Page objects to be targeted
        by uvicorn when running a script."""
        from meerkat.interactive.api import MeerkatAPI

        return MeerkatAPI

    def launch(self, return_url: bool = False):
        if state.frontend_info is None:
            rich.print("Frontend is not initialized. Running `mk.gui.start()`.")
            from .startup import start

            start()

        # TODO: restore the original route
        # We had issues using the original route when serving [slug] pages
        # in production mode, see `run_frontend_prod` in `startup.py`.
        # url = f"{state.frontend_info.url}/{self.id}"
        url = f"{state.frontend_info.url}/?id={self.id}"

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

            # in_mk_run_subprocess = int(os.environ.get("MEERKAT_RUN", 0))
            if not MEERKAT_RUN_SUBPROCESS:
                # get locals of the main module when running in script.
                import __main__

                code.interact(local=__main__.__dict__)

    @property
    def frontend(self):
        return PageFrontend(name=self.name, component=self.component.frontend)
