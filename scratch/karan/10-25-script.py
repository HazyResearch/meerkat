import meerkat as mk
import uuid
from meerkat.interactive.app.src.lib.component.sort import SortCriterion
from typing import Sequence, Union
from meerkat.interactive.graph import Box, Store, interface_op, make_store
from meerkat.ops.sliceby.groupby import groupby

ngoa = mk.get(
    'ngoa',
    '/data/datasets/opendata/',
)
ngoa_images = ngoa['published_images'].lz[:100].lz['uuid', 'image']

# The column with the unique id for the example.
ID_COLUMN = "uuid"
# The label column.
LABEL_COLUMN = "label"

ngoa_images["label"] = ["undefined"] * len(ngoa_images)


# ['uuid',
#  'iiifurl',
#  'iiifthumburl',
#  'viewtype',
#  'sequence',
#  'width',
#  'height',
#  'maxpixels',
#  'created',
#  'modified',
#  'depictstmsobjectid',
#  'assistivetext',
#  'image']

@interface_op
def groupby_and_count(data: mk.DataPanel, by: Union[str, Sequence[str]], label: str) -> mk.DataPanel:
    """Group examples by key and count the fraction of the group for this label."""
    groups = groupby(data=data, by=by)
    dp = groups.aggregate(lambda x: len(x.lz[x[LABEL_COLUMN] == label]) / len(x), accepts_dp=True)
    return mk.DataPanel({by: dp[by], label: dp["dp"]})

@interface_op
def get_dp_columns(dp: mk.DataPanel) -> Sequence[str]:
    return dp.columns

@interface_op
def get_labels(dp: mk.DataPanel) -> Sequence[str]:
    return dp["label"].unique()


ngoa_images = mk.gui.Pivot(ngoa_images)

# Filter
filter = mk.gui.Filter(ngoa_images, criteria="", title="Filter Examples")
current_examples = filter.derived()

# Match
match = mk.gui.Match(
    ngoa_images, against="image", col=ID_COLUMN, title="Search Slices"
)

# Sort
sorted_examples = mk.sort(current_examples, by=match.col, ascending=False)

# Gallery
gallery = mk.gui.Gallery(
    dp=sorted_examples,
    main_column="image",
    tag_columns=[],
    primary_key=ID_COLUMN,
)

# Label
edit_target = mk.gui.EditTarget(
    target=ngoa_images, target_id_column=ID_COLUMN, source_id_column=ID_COLUMN
)
editor = mk.gui.Editor(
    sorted_examples, col=LABEL_COLUMN,
    selected=gallery.selected,
    target=edit_target,
    primary_key=ID_COLUMN,
)

# Plot
# by = mk.gui.Choice("label", choices=get_dp_columns(ngoa_images))
# label = mk.gui.Choice("undefined", choices=get_labels(ngoa_images))
by = "label"
label = "undefined"
group_dp = groupby_and_count(current_examples, by=by, label=label)

plot = mk.gui.Plot(
    data=group_dp,
    x=label,
    y=by,
    x_label="Prevalence",
    y_label="",
    id=by,
    selection=[],
    metadata_columns=[by],
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
    },
).launch()

