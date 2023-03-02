"""Filter an image dataset in a gallery."""
import meerkat as mk

df = mk.get("imagenette", version="160px")

filter = mk.gui.Filter(df)
df = filter(df)

gallery = mk.gui.Gallery(df, main_column="img")

page = mk.gui.Page(
    mk.gui.html.flexcol([filter, gallery]),
    id="filter",
)
page.launch()
