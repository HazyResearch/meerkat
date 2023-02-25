import meerkat as mk

df = mk.get("imagenette", version="160px")

filter = mk.gui.Filter(df=df)

sort = mk.gui.Sort(df=df)

gallery = mk.gui.Gallery(df=sort(filter(df)), main_column="img", tag_columns=["label"])

mk.gui.start()
page = mk.gui.Page(
    component=mk.gui.html.grid(
        slots=[filter, sort, gallery],
    ),
    id="filter-sort",
)
page.launch()
