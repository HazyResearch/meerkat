import meerkat as mk

df = mk.get("imagenette", version="160px")

# Gallery
gallery = mk.gui.Gallery(
    df=df,
    main_column="img",
)

mk.gui.start(shareable=False)
mk.gui.Interface(component=mk.gui.RowLayout(components=[gallery])).launch()
