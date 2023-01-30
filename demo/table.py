import meerkat as mk

df = mk.get("imagenette", version="160px")

table = mk.gui.Table(df=df)

# mk.gui.start(shareable=False)
page = mk.gui.Page(component=mk.gui.html.flexcol(slots=[table]), id="table")
page.launch()
