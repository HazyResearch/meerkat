import meerkat as mk
from meerkat.interactive.app.src.lib.component.code_cell import CodeCell

df = mk.get("imagenette")

code_cell = CodeCell()
df = code_cell(df)

gallery = mk.gui.Gallery(
    df=df, 
    main_column="img",
    tag_columns=["label"],
)

network = mk.gui.start(shareable=False, dev=True)
mk.gui.Interface(
    component=mk.gui.RowLayout(components=[code_cell, gallery]),
).launch()
