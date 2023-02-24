import meerkat as mk

df = mk.get("imagenette", version="160px")

gallery = mk.gui.Gallery(df=df, main_column="img", tag_columns=["label"])

filter_criteria = mk.gui.Store([])
filter = mk.gui.Filter(df=df, criteria=filter_criteria)
df = filter(df)

sort_criteria = mk.gui.Store([])
sort = mk.gui.Sort(df=df, criteria=sort_criteria)
df = sort(df)

mk.gui.start()
page = mk.gui.Page(
    component=mk.gui.html.grid(
        slots=[filter, sort, gallery],
    ),
    id="filter-sort",
)
page.launch()
