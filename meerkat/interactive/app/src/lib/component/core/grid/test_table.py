import meerkat as mk
from meerkat.interactive import Page
from meerkat.interactive.app.src.lib.component.core.table import Table

df = mk.DataFrame(
    {
        "a": list(range(100)),
        "b": list(range(4, 104)),
    }
)
table = Table(df)

page = Page(component=table, id="table")
page.launch()
