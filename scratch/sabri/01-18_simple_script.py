
import meerkat as mk

df = mk.get("imagenette")

mk.gui.start()


df.gui.gallery(main_column="img", tag_columns="label")