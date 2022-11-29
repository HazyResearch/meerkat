import meerkat as mk

from meerkat.interactive.app.src.lib.component.mocha.test import Test
# from mocha.component.test import Test

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

gallery = Test(
    gallery=mk.gui.Gallery(
        df=df,
        main_column="img",
        tag_columns=["label"],
    )
)


mk.gui.start(shareable=False)
mk.gui.Interface(
    component=gallery,
).launch()
