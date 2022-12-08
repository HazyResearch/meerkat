import meerkat as mk

df = mk.get("imagenette", version="160px")

gallery = mk.gui.Gallery(
    df=df,
    main_column="img",
)
gallery2 = mk.gui.Gallery(
    df=df,
    main_column="img",
)
button = mk.gui.Button(title="Click me!")
button2 = mk.gui.Button(title="Click me!")
button3 = mk.gui.Button(title="Click me!")

from meerkat.interactive.app.src.lib.layouts import (
    Flex,
    Div,
    Grid,
    AutoLayout,
    ColumnLayout,
)

grid = Grid(
    components=[
        Div(component=button, classes="font-bold"),
        Div(component=button2, classes="font-black"),
        Div(component=button3, classes="w-1/3"),
    ],
)

mk.gui.start(dev=True)
mk.gui.Interface(component=ColumnLayout(components=[gallery, gallery2])).launch()
