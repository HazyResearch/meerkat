import meerkat as mk

mk.gui.start(api_port=3000)

df = mk.get("imagenette")
gallery = mk.gui.Gallery(df=df, main_column="img")
# gallery = mk.gui.Gallery(df=df, main_column="img", tag_columns=["path"])

page = mk.gui.Page(component=gallery, id="page")
page.launch()