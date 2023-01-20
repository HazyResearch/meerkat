import meerkat as mk

df = mk.get("imagenette")


gallery = mk.gui.Gallery(df=df, main_column="img", tag_columns=["label"])

interface = mk.gui.Interface(component=gallery, id="simple")
interface.launch()
