import meerkat as mk

# df = mk.get("imagenette", version="160px").lz[:200]

df = mk.get("imdb", registry="huggingface")  # pull text data from huggingface example
df = df["train"]

df["text"] = df["text"].to_pandas()
df["label"] = df["label"].to_pandas()

# df = mk.get("ngoa")["published_images"].lz[:100] # national gallery of art multimodal data example
# df = mk.get("coco", version="2014", download_mode="force").lz[:200] # pull ms-coco multimodal data example

# emb_df = mk.DataFrame.read(
#     "ngoa_published_images_224_clip.mk/"
# )
# images = ngoa["published_images"].merge(emb_df, on="uuid")

df_pivot = mk.gui.Reference(df)

df = mk.embed(
    df,
    input="text",
    batch_size=128,
)

match: mk.gui.Component = mk.gui.Match(df_pivot, against="text", col="label")

sorted_box = mk.sort(df_pivot, by=match.col, ascending=False)

gallery = mk.gui.Gallery(
    sorted_box, main_column="text", tag_columns=["label"], primary_key="id"
)

mk.gui.start(shareable=False)
mk.gui.Interface(components=[match, gallery]).launch()
