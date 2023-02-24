import meerkat as mk

df = mk.get("imagenette", version="160px")

filter = mk.gui.Filter(df=df)
df = filter(df)

criteria = mk.gui.Store([])
sort = mk.gui.Sort(df=df, criteria=criteria)
df = sort(df)

gallery = mk.gui.Gallery(df=df, main_column="img")

mk.gui.start()
page = mk.gui.Page(
    component=mk.gui.html.div(
        slots=[filter, sort, gallery],
        classes="flex flex-row",
    ),
    id="filter-sort",
)
page.launch()
