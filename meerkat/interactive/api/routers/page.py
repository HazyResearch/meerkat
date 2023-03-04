from fastapi.encoders import jsonable_encoder

from meerkat.interactive import Page
from meerkat.interactive.endpoint import endpoint
from meerkat.interactive.utils import get_custom_json_encoder
from meerkat.tools.lazy_loader import LazyLoader

torch = LazyLoader("torch")


@endpoint(prefix="/page", route="/{page}/config/", method="GET")
def config(page: Page):
    return jsonable_encoder(
        # TODO: we should not be doing anything except page.frontend
        # here. This is a temp workaround to avoid getting an
        # exception in the notebook.
        page.frontend if isinstance(page, Page) else page,
        custom_encoder=get_custom_json_encoder(),
    )
