import meerkat as mk

df = mk.get("imagenette",)[:100]

# dp = mk.embed(
#     dp,
#     input="img",
#     batch_size=128,
# )

match: mk.gui.Component = mk.gui.FormulaBar(
    df=df,
    against="img",
    col="label"
)

# sorted_box = mk.sort(dp_pivot, by=match.col, ascending=False)

# gallery = mk.gui.Gallery(
#     sorted_box,
#     main_column="img",
#     tag_columns=["label"],
# )

mk.gui.start()
mk.gui.Interface(
    match
).launch()
