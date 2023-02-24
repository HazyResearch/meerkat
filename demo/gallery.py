import meerkat as mk

df = mk.get("imagenette", version="160px")
gallery = mk.gui.Gallery(df, main_column="img")

page = mk.gui.Page(mk.gui.html.div(gallery, classes="h-[600px]"), id="gallery")
page.launch()
