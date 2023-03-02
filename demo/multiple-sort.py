"""Multiple galleries in tabs, with independent sorting."""
import meerkat as mk

df1 = mk.get("imagenette", version="160px")
df2 = mk.get("imagenette", version="160px")

sort1 = mk.gui.Sort(df1, title="Sort (Dataframe 1)")
sort2 = mk.gui.Sort(df2, title="Sort (Dataframe 2)")

galleries = mk.gui.Tabs(
    tabs={
        "df1": mk.gui.Gallery(sort1(df1), main_column="img"),
        "df2": mk.gui.Gallery(sort2(df2), main_column="img"),
    }
)

page = mk.gui.Page(
    mk.gui.html.flexcol(slots=[sort1, sort2, galleries]),
    id="multiple-sort",
)
page.launch()
