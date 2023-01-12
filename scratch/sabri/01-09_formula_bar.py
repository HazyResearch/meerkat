import meerkat as mk
from meerkat.interactive.app.src.lib.component.formula_bar import FormulaBar

df = mk.get("imagenette")

formula_bar = FormulaBar()
df = formula_bar(df)

gallery = mk.gui.Gallery(
    df=df, 
    main_column="img",
    tag_columns=["label"],
)

network = mk.gui.start(shareable=False, dev=True)
mk.gui.Interface(
    component=mk.gui.RowLayout(components=[formula_bar, gallery]),
).launch()
