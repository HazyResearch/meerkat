from meerkat.interactive import Interface, endpoint
from meerkat.interactive.app.src.lib.component.button import Button


@endpoint
def on_click():
    # Must use flush=True to flush the stdout buffer
    print("Button clicked", flush=True)


button = Button(
    title="Button",
    on_click=on_click,
)

interface = Interface(component=button, id="button")
interface.launch()
