import meerkat as mk


mk.gui.start(api_port=3000)

df = mk.get("imagenette")


gallery = mk.gui.Gallery(df=df, main_column="img", tag_columns=["path"])

interface = mk.gui.Interface(component=gallery, id="simple")
interface.launch()


