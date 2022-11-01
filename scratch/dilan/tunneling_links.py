import meerkat as mk

dp = mk.get("imagenette", version="160px")

dp_pivot = mk.gui.Pivot(dp)

value="Hello World"

gallery = mk.gui.Gallery(
    dp_pivot,
    main_column="img",
    tag_columns=["label"]
)

markdown = mk.gui.Markdown(
    """## Hello World"""
    """I just love **bold text**."""
    """Italicized text is the *cat's meow*."""
)

mk.gui.start()
mk.gui.Interface(
    components=[markdown, gallery],
).launch()