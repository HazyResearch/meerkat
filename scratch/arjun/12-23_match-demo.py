import os
import meerkat as mk

IMAGE_COLUMN = "img"
EMBED_COLUMN = "img_clip"

path = "~/.meerkat/dataframes/imagenette_clip.mk"
path = os.path.abspath(os.path.expanduser(path))
if not os.path.exists(path):
    df = mk.get("imagenette", version="160px")

    # Embed the image.
    df: mk.DataFrame = mk.embed(df, input=IMAGE_COLUMN, out_col=EMBED_COLUMN)
    df.write("~/.meerkat/dataframes/imagenette_clip.mk")
else:
    df = mk.DataFrame.read(path)

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
