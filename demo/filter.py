import meerkat as mk

df = mk.get("imagenette", version="160px")

filter = mk.gui.Filter(df)
df = filter(df)

# Gallery
gallery = mk.gui.Gallery(
    df=df,
    main_column="img",
)

page = mk.gui.Page(
    component=mk.gui.html.flexcol(slots=[filter, gallery]),
    id="filter",
)
page.launch()
