import meerkat as mk

df = mk.get("imagenette", version="160px")

# Gallery
gallery = mk.gui.Gallery(
    df=df,
    main_column="img",
)

mk.gui.start(shareable=False)
page = mk.gui.Page(component=mk.gui.html.flexcol(slots=[gallery]), id="gallery")
page.launch()
