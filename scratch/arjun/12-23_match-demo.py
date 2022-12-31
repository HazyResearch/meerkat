import meerkat as mk

IMAGE_COLUMN = "img"
EMBED_COLUMN = "img_clip"

df = mk.get("imagenette", version="160px")

# Embed the image.
df = mk.embed(df, input=IMAGE_COLUMN, out_col=EMBED_COLUMN)

with mk.gui.react():
    # Match
    match = mk.gui.Match(df=df, against=EMBED_COLUMN)
    examples_df = match(df)[0]

    # Sort
    df_sorted = mk.sort(data=examples_df, by=match.criterion.name, ascending=False)

# Gallery
gallery = mk.gui.Gallery(
    df=df_sorted,
    main_column=IMAGE_COLUMN,
)

mk.gui.start(shareable=False)
mk.gui.Interface(component=mk.gui.RowLayout(components=[match, gallery])).launch()
