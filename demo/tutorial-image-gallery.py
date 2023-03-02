"""Display images in an interactive gallery.

This is a tutorial on how to build a basic interactive application with
Meerkat.
"""
import meerkat as mk

df = mk.get("imagenette", version="160px")
gallery = mk.gui.Gallery(df, main_column="img", tag_columns=["path", "label"])

page = mk.gui.Page(gallery, id="gallery")
page.launch()
