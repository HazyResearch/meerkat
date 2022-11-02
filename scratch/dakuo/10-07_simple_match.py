
import meerkat as mk

# dp = mk.get("imagenette", version="160px").lz[:200]

dp = mk.get("imdb", registry="huggingface") # pull text data from huggingface example
dp = dp['train']

dp['text'] = dp['text'].to_pandas()
dp['label'] = dp['label'].to_pandas()

# dp = mk.get("ngoa")["published_images"].lz[:100] # national gallery of art multimodal data example
# dp = mk.get("coco", version="2014", download_mode="force").lz[:200] # pull ms-coco multimodal data example

# emb_dp = mk.DataPanel.read(
#     "ngoa_published_images_224_clip.mk/"
# )
# images = ngoa["published_images"].merge(emb_dp, on="uuid")

dp_pivot = mk.gui.Pivot(dp)

dp = mk.embed(
    dp,
    input="text",
    batch_size=128,
)

match: mk.gui.Component = mk.gui.Match(
    dp_pivot, 
    against="text",
    col="label"
)

sorted_box = mk.sort(dp_pivot, by=match.col, ascending=False)

gallery = mk.gui.Gallery(
    sorted_box,
    main_column="text",
    tag_columns=["label"],
    primary_key="id"
)

mk.gui.start(shareable=False)
mk.gui.Interface(
    components=[match, gallery]
).launch()