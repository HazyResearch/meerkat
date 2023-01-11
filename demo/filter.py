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
mk.gui.Interface(component=mk.gui.RowLayout(components=[filter, gallery])).launch()
