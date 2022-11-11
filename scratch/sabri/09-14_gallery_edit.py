import numpy as np

import meerkat as mk

df = mk.get("imagenette", version="160px")
df["car"] = np.zeros(len(df))
df = mk.gui.Reference(df)


@mk.gui.interface_op
def filter(df: mk.DataFrame):
    return df.lz[df["label"] == "gas pump"]


filtered_df = filter(df)

target = mk.gui.EditTarget(
    target=df, target_id_column="img_path", source_id_column="img_path"
)

gallery = mk.gui.Gallery(
    df=filtered_df,
    main_column="img",
    tag_columns=["label", "car"],
    edit_target=target,
)

editor = mk.gui.Editor(
    df=filtered_df, target=target, col="car", selected=gallery.selected
)

mk.gui.start()
mk.gui.Interface(components=[editor, gallery]).launch()
