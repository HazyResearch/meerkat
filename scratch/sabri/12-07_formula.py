import meerkat as mk

df = mk.get(
    "imagenette",
).lz[:100]

# dp = mk.embed(
#     dp,
#     input="img",
#     batch_size=128,
# )

bar: mk.gui.Component = mk.gui.FormulaBar(df=df, against="img", col="label")

# sorted_box = mk.sort(dp_pivot, by=bar.col, ascending=False)

# gallery = mk.gui.Gallery(
#     sorted_box,
#     main_column="img",
#     tag_columns=["label"],
# )

mk.gui.start(dev=True)
mk.gui.Interface(component=bar).launch()
