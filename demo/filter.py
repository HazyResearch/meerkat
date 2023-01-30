import meerkat as mk

df = mk.get("imagenette", version="160px")

with mk.gui.react():
    filter = mk.gui.Filter(df=df)
    df = filter(df)

# Gallery
gallery = mk.gui.Gallery(
    df=df,
    main_column="img",
)

mk.gui.start(shareable=False)
interface = mk.gui.Page(
    component=mk.gui.html.flexcol(slots=[filter, gallery]),
    id="filter",
)
interface.launch()
