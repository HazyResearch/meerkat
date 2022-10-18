
import meerkat as mk

dp = mk.get("imagenette").lz[:1000]
dp_pivot = mk.gui.Pivot(dp)

dp = mk.embed(
    dp,
    input="img",
    batch_size=128,
)

match: mk.gui.Component = mk.gui.Match(
    dp_pivot, 
    against="img",
    col="label"
)

sorted_box = mk.sort(dp_pivot, by=match.col, ascending=False)

gallery = mk.gui.Gallery(
    sorted_box,
    main_column="img",
    tag_columns=["label"],
)

mk.gui.start()
mk.gui.Interface(
    components=[match, gallery]
).launch()