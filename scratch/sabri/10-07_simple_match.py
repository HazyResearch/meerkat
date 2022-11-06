import meerkat as mk

df = mk.get("imagenette", version="160px").lz[:2000]
df_pivot = mk.gui.Pivot(df)

df = mk.embed(
    df,
    input="img",
    batch_size=128,
    encoder="clip"
)

match: mk.gui.Component = mk.gui.Match(df_pivot, against="img", col="label")

sorted_box = mk.sort(df_pivot, by=match.col, ascending=False)

gallery = mk.gui.Gallery(
    sorted_box, main_column="img", tag_columns=["label"], primary_key="img_path"
)

mk.gui.start(shareable=False)
mk.gui.Interface(
    components=[match, gallery]
).launch()

