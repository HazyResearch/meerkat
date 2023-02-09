from typing import Any

from meerkat.interactive import Page, endpoint, print
from meerkat.interactive.app.src.lib.component.core.toggle import Toggle


@endpoint
def on_change(value: Any):
    # Must use flush=True to flush the stdout buffer
    print(f"Toggled {value}", flush=True)


toggle = Toggle(
    on_change=on_change,
)

print(toggle.value)

page = Page(component=toggle, id="toggle")
page.launch()
