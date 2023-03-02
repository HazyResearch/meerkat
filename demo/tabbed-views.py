"""Display a data frame in an interactive table and gallery in a tabbed
view."""
import meerkat as mk

# Load in the imagenette dataset
df = mk.get("imagenette", version="160px")

# Create a Table and Gallery view of the data
table = mk.gui.Table(df)
gallery = mk.gui.Gallery(df, main_column="img")

# Put the two views in a Tabs component
tabs = mk.gui.Tabs(tabs={"Table": table, "Gallery": gallery})

# Show the page!
page = mk.gui.Page(
    mk.gui.html.flexcol([tabs], classes="h-full"),
    id="display",
)
page.launch()
