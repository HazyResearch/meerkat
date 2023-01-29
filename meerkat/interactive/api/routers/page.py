from meerkat.interactive import Page
from meerkat.interactive.endpoint import endpoint


@endpoint(prefix="/page", route="/{page}/config/", method="GET")
def config(page: Page):
    return page.frontend
