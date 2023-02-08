import functools

import meerkat as mk
from meerkat.interactive import Page
from meerkat.interactive.app.src.lib.component.core.raw_html import RawHTML

rawhtml = functools.partial(
    RawHTML,
    html="""\
<div>
    <p>Some text</p>
    <p>Some more text</p>
</div>

<div>
    <p>Some text</p>
</div>
"""
)


rawhtml_full = rawhtml(view="full")
rawhtml_thumbnail = rawhtml(view="thumbnail")
rawhtml_logo = rawhtml(view="logo")
component = mk.gui.html.div(slots=[
    rawhtml_full,
    rawhtml_thumbnail,
    rawhtml_logo,
])

page = Page(component=component, id="rawhtml")
page.launch()
