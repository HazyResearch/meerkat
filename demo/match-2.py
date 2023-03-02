"""Similarity search with CLIP in an interactive image gallery, with advanced
controls for user feedback."""
import rich

import meerkat as mk
from meerkat.interactive.app.src.lib.component.contrib import GalleryQuery

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
    "https://huggingface.co/datasets/meerkat-ml/meerkat-dataframes/resolve/main/embeddings/imagenette_160px.mk.tar.gz",  # noqa: E501
    overwrite=False,
)
df = df.merge(df_clip[["img_id", "img_clip"]], on="img_id")

page = mk.gui.Page(
    GalleryQuery(df, main_column=IMAGE_COLUMN, against=EMBED_COLUMN),
    id="match",
)
page.launch()
