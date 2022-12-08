import meerkat as mk

df = mk.get("imagenette", version="160px")


df_pivot = mk.gui.Reference(df)

gallery = mk.gui.Gallery(
    df_pivot,
    main_column="img",
    tag_columns=["label"],
)

mk.gui.start(shareable=True)
mk.gui.Interface(
    components=[gallery],
).launch()
