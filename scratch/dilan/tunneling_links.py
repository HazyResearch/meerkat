import meerkat as mk

df = mk.get("imagenette", version="160px")

df_pivot = mk.gui.Reference(df)

value = "Hello World"

gallery = mk.gui.Gallery(df_pivot, main_column="img", tag_columns=["label"])

markdown = mk.gui.Markdown(
    """## Hello World"""
    """I just love **bold text**."""
    """Italicized text is the *cat's meow*."""
)

mk.gui.start()
mk.gui.Interface(
    components=[markdown, gallery],
).launch()
