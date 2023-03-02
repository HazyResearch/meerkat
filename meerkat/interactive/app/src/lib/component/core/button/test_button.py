from meerkat.interactive import Page, endpoint
from meerkat.interactive.app.src.lib.component.core.button import Button


@endpoint
def on_click():
    # Must use flush=True to flush the stdout buffer
    print("Button clicked", flush=True)


button = Button(
    title="Button",
    on_click=on_click,
)

page = Page(component=button, id="button")
page.launch()
