import meerkat as mk

df = mk.get("imagenette")

tabs = mk.gui.Tabs(
    tabs={
        label: mk.gui.Gallery(
            df=df.lz[df["label"] == label],
            main_column="img",
            tag_columns=["label"],
        )
        for label in df["label"].unique()
    }
)


mk.gui.start(shareable=False, dev=False)
mk.gui.Interface(
    component=tabs,
).launch()
