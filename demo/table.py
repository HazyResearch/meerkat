import meerkat as mk

df = mk.get("imagenette", version="160px")

table = mk.gui.Table(df=df)

mk.gui.start(shareable=False)
mk.gui.Page(component=mk.gui.RowLayout(components=[table])).launch()
