import meerkat as mk

df = mk.get("imagenette", version="160px")

filter = mk.gui.Filter(df=df)
df = filter(df)

criteria = mk.Store([])
sort = mk.gui.Sort(df=df, criteria=criteria)
df = sort(df)

# Gallery
gallery = mk.gui.Gallery(
    df=df,
    main_column="img",
)

mk.gui.start(shareable=False)
page = mk.gui.Page(
    component=mk.gui.html.grid(slots=[filter, sort, gallery]), id="filter-sort"
)
page.launch()
