import meerkat as mk
import rich

IMAGE_COLUMN = "img"
EMBED_COLUMN = "img_clip"

rich.print(
    "[bold red]This script uses CLIP to embed images. "
    "This will take some time to download the model. "
    "Please be patient.[/bold red]"
)

df = mk.get("imagenette", version="160px")
# Download the precomupted CLIP embeddings for imagenette.
# You can also embed the images yourself with mk.embed. This will take some time.
# To embed: df = mk.embed(df, input=IMAGE_COLUMN, out_col=EMBED_COLUMN, encoder="clip").
df_clip = mk.DataFrame.read(
    "https://huggingface.co/datasets/arjundd/meerkat-dataframes/resolve/main/embeddings/imagenette_160px.mk.tar.gz",  # noqa: E501
    overwrite=False,
)
df_clip = df_clip[["img_id", "img_clip"]]
df = df.merge(df_clip, on="img_id")

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
page = mk.gui.Page(
    component=mk.gui.RowLayout(slots=[match, gallery]), id="match"
)
page.launch()
