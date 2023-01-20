import meerkat as mk

df = mk.get("imagenette", version="160px")

with mk.gui.react():
    filter = mk.gui.Filter(df=df)
    df = filter(df)

    criteria = mk.gui.Store([])
    sort = mk.gui.Sort(df=df, criteria=criteria)
    df = sort(df)

# Gallery
gallery = mk.gui.Gallery(
    df=df,
    main_column="img",
)

mk.gui.start(shareable=False)
interface = mk.gui.Interface(
    component=mk.gui.RowLayout(slots=[filter, sort, gallery]), id="filter-sort"
)
interface.launch()
