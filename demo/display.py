import meerkat as mk

df = mk.get("imagenette", version="160px")


table = mk.gui.Table(df=df)
gallery = mk.gui.Gallery(df=df, main_column="img")
tabs = mk.gui.Tabs(tabs={"Table": table, "Gallery": gallery})

mk.gui.start(shareable=False)
mk.gui.Interface(component=mk.gui.RowLayout(components=[tabs])).launch()
