"""Sort an image dataset in a gallery."""
import meerkat as mk

df = mk.get("imagenette", version="160px")
df = df.mark()

sort = mk.gui.Sort(df)

# Show the sorted DataFrame in a gallery.
gallery = mk.gui.Gallery(sort(df), main_column="img")

page = mk.gui.Page(
    mk.gui.html.div([sort, gallery], classes="flex flex-col"),
    id="sort",
)
page.launch()
