import numpy as np

import meerkat as mk

dp = mk.get("imagenette")
dp["car"] = np.zeros(len(dp))
dp = mk.gui.Pivot(dp)


@mk.gui.interface_op
def filter(dp: mk.DataPanel):
    return dp.lz[dp["label"] == "gas pump"]


filtered_dp = filter(dp)

editor = mk.gui.Editor(
    dp=filtered_dp,
    edit_target=mk.gui.EditTarget(
        pivot=dp, pivot_id_column="img_path", id_column="img_path"
    ),
)

gallery = mk.gui.Gallery(
    dp=filtered_dp,
    main_column="img",
    tag_columns=["label"],
    edit_target=mk.gui.EditTarget(
        pivot=dp, pivot_id_column="img_path", id_column="img_path"
    ),
)

mk.gui.start()
mk.gui.Interface(components=[editor, gallery]).launch()
