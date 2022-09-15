import numpy as np

import meerkat as mk

dp = mk.get("imagenette")
dp["car"] = np.zeros(len(dp))
dp = mk.gui.Pivot(dp)


@mk.gui.interface_op
def filter(dp: mk.DataPanel):
    return dp.lz[dp["label"] == "gas pump"]


filtered_dp = filter(dp)

target = mk.gui.EditTarget(
    target=dp, target_id_column="img_path", source_id_column="img_path"
)

gallery = mk.gui.Gallery(
    dp=filtered_dp,
    main_column="img",
    tag_columns=["label"],
    edit_target=target,
)

editor = mk.gui.Editor(
    dp=filtered_dp,
    target=target,
    col="car",
    selected=gallery.selected
)

mk.gui.start()
mk.gui.Interface(components=[editor, gallery]).launch()
