import meerkat as mk

df = mk.get("imagenette", version="160px")
assert isinstance(df, mk.DataFrame)

gallery = mk.gui.Gallery(
    df=df,
    main_column="img",
)

page = mk.gui.Page(component=mk.gui.html.flexcol(gallery), id="gallery")
page.launch()
