"""Test that sort1 and sort2 are independent of each other."""
import meerkat as mk

with mk.gui.react():
    df1 = mk.get("imagenette", version="160px")
    df2 = mk.get("imagenette", version="160px")

    sort1 = mk.gui.Sort(df=df1)
    sort2 = mk.gui.Sort(df=df2)

    galleries = mk.gui.Tabs(
        tabs={
            "df1": mk.gui.Gallery(df=sort1(df1), main_column="img"),
            "df2": mk.gui.Gallery(df=sort2(df2), main_column="img"),
        }
    )

mk.gui.start(shareable=False)
page = mk.gui.Page(
    component=mk.gui.html.flexcol(slots=[sort1, sort2, galleries]), 
    id="multiple-sort",
)
page.launch()
