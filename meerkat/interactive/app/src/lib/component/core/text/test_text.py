import meerkat as mk
from meerkat.interactive import Page, print
from meerkat.interactive.app.src.lib.component.core.text import Text

text_1 = Text(
    data="Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed euismod, nisl vitae ultricies lacinia, nisl nisl aliquam nisl, "
    "vitae aliquam nisl nisl sit amet nisl. Nulla",
)
text_2 = Text(
    data="Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed euismod, nisl vitae ultricies lacinia, nisl nisl aliquam nisl, "
    "vitae aliquam nisl nisl sit amet nisl. Nulla",
    editable=True,
)
text_3 = Text(
    data="Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed euismod, nisl vitae ultricies lacinia, nisl nisl aliquam nisl, "
    "vitae aliquam nisl nisl sit amet nisl. Nulla",
    view="wrapped",
)

print(text_1.data)
print(text_2.data)
component = mk.gui.html.flexcol(
    [text_1, text_2, text_3]
)

page = Page(component=component, id="text")
page.launch()
