import meerkat as mk

df = mk.get("imagenette")

tabs = mk.gui.Tabs(
    tabs={
        label: mk.gui.Gallery(
            df=df[df["label"] == label],
            main_column="img",
            tag_columns=["label"],
        )
        for label in df["label"].unique()
    }
)
gallery = mk.gui.Gallery(
    df=df, 
    main_column="img",
    tag_columns=["label"],
)


network = mk.gui.start(shareable=False, dev=True)
mk.gui.Interface(
    component=gallery,
).launch()
