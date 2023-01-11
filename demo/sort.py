import meerkat as mk

df = mk.get("imagenette", version="160px")

with mk.gui.react():
    criteria = mk.gui.Store([])
    sort = mk.gui.Sort(df=df, criteria=criteria)
    df = sort(df)

# Gallery
gallery = mk.gui.Gallery(
    df=df,
    main_column="img",
)

mk.gui.start(shareable=False)
mk.gui.Interface(component=mk.gui.RowLayout(components=[sort, gallery])).launch()
