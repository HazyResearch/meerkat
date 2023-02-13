import meerkat as mk
from meerkat.interactive import react

df = mk.get("imagenette", version="160px")
df = react(df)
assert isinstance(df, mk.DataFrame)

sort = mk.gui.Sort(df)

# Show the sorted DataFrame in a gallery.
gallery = mk.gui.Gallery(sort(df), main_column="img")

page = mk.gui.Page(
    component=mk.gui.html.flexcol([sort, gallery]),
    id="sort",
)
page.launch()
