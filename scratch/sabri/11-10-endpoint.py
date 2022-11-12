import meerkat as mk
from meerkat.interactive.graph import endpoint


df = mk.get("imagenette", version="160px").lz[:2000]
df_ref = mk.gui.Reference(df)

# df = mk.embed(
#     df,
#     input="img",
#     batch_size=128,
#     encoder="clip"
# )

# match: mk.gui.Component = mk.gui.Match(df_ref, against="img", col="label")

# sorted_box = mk.sort(df_ref, by=match.col, ascending=False)


@endpoint
def change_tag_columns(columns: mk.gui.Store):
    if len(columns._) == 0:
        columns._ = ["label"]
    else:
        columns._ = []


tag_columns = mk.gui.Store(["label"])

gallery = mk.gui.Gallery(
    df_ref, main_column="img", tag_columns=tag_columns, primary_key="img_path"
)

button = mk.gui.Button(on_click=change_tag_columns(tag_columns))

mk.gui.start()
mk.gui.Interface(components=[gallery, button]).launch()
