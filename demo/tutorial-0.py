import meerkat as mk

df = mk.get("imagenette", version="160px")
mk.gui.start(api_port=3000)

gallery = mk.gui.Gallery(df=df, main_column="img", tag_columns=["path", "label"])

page = mk.gui.Page(component=gallery, id="gallery")
page.launch()
