import meerkat as mk

IMAGE_COLUMN = "img"
EMBED_COLUMN = "img_clip"

df = mk.get("imagenette", dataset_dir="/Users/arjundd/.cache/meerkat/datasets", version="160px")
df = df[100:150]

# Embed the image.
df = mk.embed(df, input=IMAGE_COLUMN, out_col=EMBED_COLUMN)

with mk.gui.react():
    # Match
    match = mk.gui.Match(
        df=df,
        against=EMBED_COLUMN,
        title="Search Examples",
    )
    examples_df = match(df)

    # Sort
    # df_sorted = mk.sort(data=examples_df, by=match.criterion.name, ascending=False)[0]

# Gallery
gallery = mk.gui.Gallery(
    df=df,
    main_column=IMAGE_COLUMN,
)

mk.gui.start(shareable=False)
mk.gui.Interface(
    component=mk.gui.RowLayout(components=[match, gallery])
).launch()
