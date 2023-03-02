"""Display a data frame in an interactive table."""
import meerkat as mk

df = mk.get("imagenette", version="160px")

table = mk.gui.Table(df)

page = mk.gui.Page(table, id="table")
page.launch()
