import functools

import meerkat as mk
from meerkat.interactive import Page
from meerkat.interactive.app.src.lib.component.core.pdf import PDF

rawhtml = functools.partial(PDF, data="https://arxiv.org/pdf/0704.0001.pdf")


rawhtml_full = rawhtml(view="full")
component = mk.gui.html.div(
    slots=[
        rawhtml_full,
    ]
)

page = Page(component=component, id="rawhtml")
page.launch()
