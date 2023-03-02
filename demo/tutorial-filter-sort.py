import meerkat as mk

df = mk.get("imagenette", version="160px")

filter = mk.gui.Filter(df)
sort = mk.gui.Sort(df)
gallery = mk.gui.Gallery(sort(filter(df)), main_column="img", tag_columns=["label"])

page = mk.gui.Page(
    mk.gui.html.grid([filter, sort, gallery], classes="h-screen grid grid-rows-[auto,auto,1fr] p-3"),
    id="filter-sort",
)
page.launch()
