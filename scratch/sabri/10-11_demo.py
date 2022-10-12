
import meerkat as mk

dp = mk.get("imagenette", version="160px")


dp_pivot = mk.gui.Pivot(dp)

gallery = mk.gui.Gallery(
    dp_pivot,
    main_column="img",
    tag_columns=["label"],
)

mk.gui.start()
mk.gui.Interface(
    components=[gallery],
).launch()
