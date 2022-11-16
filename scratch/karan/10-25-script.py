import uuid
from glob import glob
from typing import Sequence, Union

import numpy as np
import pandas as pd

import meerkat as mk
from meerkat.interactive.app.src.lib.component.sort import SortCriterion
from meerkat.interactive.graph import interface_op
from meerkat.ops.sliceby.groupby import groupby

#### New Dataloading

# Load National Gallery of Art (NGOA) tables
ngoa = mk.get("ngoa", "/data/datasets/opendata/")
published_images, objects, constituents, objects_constituents = (
    ngoa["published_images"],
    ngoa["objects"],
    ngoa["constituents"],
    ngoa["objects_constituents"],
)
objects = objects.merge(
    objects_constituents["objectid", "constituentid"],
    left_on="objectid",
    right_on="objectid",
).merge(
    constituents["constituentid", "visualbrowsernationality"],
    left_on="constituentid",
    right_on="constituentid",
)

# Load NGOA published images at 224 x 224 (rescaled to fit this size) and CLIP embeddings
clip_df = mk.DataFrame(
    {"image_path": glob("/data/datasets/opendata/published_images_224/*")}
)
clip_df["uuid"] = clip_df["image_path"].str.split("/").str.get(-1).str.rstrip(".jpg")
clip_df["image_224"] = mk.ImageColumn.from_filepaths(clip_df["image_path"])
clip_df = clip_df.merge(
    mk.DataFrame.read("/data/datasets/opendata/ngoa_published_images_224_clip.mk/"),
    on="uuid",
)
clip_df = clip_df.sort("uuid")

# Merge the images and embeddings into the published_images table
published_images = mk.merge(published_images, clip_df, on="uuid")
# Merge constituents info.
# published_images = mk.merge(published_images, constituents, left_on="attribution")

# Merge the published_images and objects tables, along with embeddings saved on disk for the text columns in the objects
df = mk.merge(
    published_images, objects, left_on="depictstmsobjectid", right_on="objectid"
)
df = df[
    "objectid",
    "uuid",
    "image_224",
    "clip(image_224)",
    "title",
    "parentid",
    "beginyear",
    "endyear",
    "medium",
    "dimensions",
    "inscription",
    "attribution",
    "creditline",
    "visualbrowserclassification",
    "visualbrowsernationality",
]

df["medium"] = df["medium"].astype("str")
df["dimensions"] = df["dimensions"].astype("str")
df["inscription"] = df["inscription"].astype("str")
df["attribution"] = df["attribution"].astype("str")
df["creditline"] = df["creditline"].astype("str")
df["title"] = df["title"].astype("str")

# Merge embeddings for the text columns
ngoa_images = df.merge(
    mk.DataFrame.read(
        "/data/datasets/opendata/ngoa_objects_images_clip_embs.mk/",
    ),
    on="objectid",
)
# Filter duplicates
ngoa_images = ngoa_images.lz[~ngoa_images["uuid"].to_pandas().duplicated()]

# ngoa = mk.get(
#     "ngoa",
#     "/data/datasets/opendata/",
# )
# ngoa_images = ngoa["published_images"].lz[:100]#.lz["uuid", "image"]


def filter_na(df: mk.DataFrame):
    for name in df.columns:
        col = df[name]
        is_pd_na = isinstance(col, mk.PandasSeriesColumn) and np.any(pd.isna(col.data))
        is_np_na = isinstance(col, mk.NumpyArrayColumn) and np.any(np.isnan(col.data))
        if is_pd_na or is_np_na:
            df.remove_column(name)
            print(name)
    return df


# Filter out nans.
ngoa_images = filter_na(ngoa_images)

####

# The column with the unique id for the example.
ID_COLUMN = "uuid"
# Image Column.
IMAGE_COLUMN = "image_224"
# The label column.
LABEL_COLUMN = "label"

ngoa_images["label"] = ["undefined"] * len(ngoa_images)


@interface_op
def groupby_and_count(
    data: mk.DataFrame, by: Union[str, Sequence[str]], label: str
) -> mk.DataFrame:
    """Group examples by key and count the fraction of the group for this label."""
    data = mk.DataFrame({by: data[by], f"is_{label}": data[LABEL_COLUMN] == label})
    groups = groupby(data=data, by=by).mean()
    return mk.DataFrame({"group": groups[by], "prevalence": groups[f"is_{label}"]})


@interface_op(nested_return=False)
def get_df_columns(df: mk.DataFrame) -> Sequence[str]:
    return df.columns


@interface_op(nested_return=False)
def get_labels(df: mk.DataFrame) -> Sequence[str]:
    return df["label"].unique().tolist()


ngoa_images = mk.gui.Reference(ngoa_images)

# Match
match = mk.gui.Match(
    ngoa_images, against=IMAGE_COLUMN, col=ID_COLUMN, title="Search Slices"
)

# Filter
filter = mk.gui.Filter(ngoa_images, criteria="", title="Filter Examples")
current_examples = filter.derived()

# Sort
sorted_examples = mk.sort(current_examples, by=match.col, ascending=False)

# Gallery
gallery = mk.gui.Gallery(
    df=sorted_examples,
    main_column=IMAGE_COLUMN,
    tag_columns=[],
    primary_key=ID_COLUMN,
)

# Label
edit_target = mk.gui.EditTarget(
    target=ngoa_images, target_id_column=ID_COLUMN, source_id_column=ID_COLUMN
)
editor = mk.gui.Editor(
    sorted_examples,
    col=LABEL_COLUMN,
    selected=gallery.selected,
    target=edit_target,
    primary_key=ID_COLUMN,
)

# Plot
var1 = get_df_columns(ngoa_images)
var2 = get_labels(ngoa_images)
print(var1)
print(var2)
by = mk.gui.Choice("label", choices=var1)
label = mk.gui.Choice("undefined", choices=var2)
# by = "label"
# label = "undefined"
group_df = groupby_and_count(current_examples, by=by.value, label=label.value)

plot = mk.gui.Plot(
    data=group_df,
    x="prevalence",
    y="group",
    x_label="Prevalence",
    y_label="",
    id="group",
    selection=[],
    metadata_columns=[],
    can_remove=False,
)

mk.gui.start(shareable=True)
mk.gui.Interface(
    layout=mk.gui.Layout("HazyDemo"),
    components={
        "gallery": gallery,
        "match": match,
        "filter": filter,
        "editor": editor,
        "gallery": gallery,
        "plot": plot,
        "groupby": by,
        "labelby": label,
    },
).launch()
