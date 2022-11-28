import meerkat as mk

from mocha.component.test import Test

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

# gallery = Test()


mk.gui.start(shareable=False)
mk.gui.Interface(
    component=tabs,
).launch()
