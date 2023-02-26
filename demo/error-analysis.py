from typing import List

import meerkat as mk
from meerkat.interactive import html
from meerkat.interactive.app.src.lib.component.core.filter import FilterCriterion

# Data loading.
imagenette = mk.get("imagenette", version="160px")
# df = mk.DataFrame.read(
#     "https://huggingface.co/datasets/meerkat-ml/meerkat-dataframes/resolve/main/imagenette-160px-facebook-convnext-tiny-224.mk.tar.gz",
#     overwrite=False,
# )
df = mk.DataFrame.read("~/Downloads/imagenette-remapped.mk")
df = imagenette.merge(df[["img_id", "logits", "pred"]], on="img_id")

# Download precomupted CLIP embeddings for imagenette.
df_clip = mk.DataFrame.read(
    "https://huggingface.co/datasets/meerkat-ml/meerkat-dataframes/resolve/main/embeddings/imagenette_160px.mk.tar.gz",  # noqa: E501
    overwrite=False,
)
df_clip = df_clip[["img_id", "img_clip"]]
df = df.merge(df_clip, on="img_id")
df["correct"] = df.map(lambda label, pred: label in pred, batch_size=len(df), pbar=True)
df.create_primary_key(column="pkey")

IMAGE_COLUMN = "img"
EMBED_COLUMN = "img_clip"

df.mark()
match = mk.gui.Match(df, against=EMBED_COLUMN)
filter = mk.gui.Filter(df)
df = filter(df)


@mk.reactive()
def add_match_criterion_to_sort(match_criterion, sort_criteria):
    # make a copy of the sort criteria.
    sort_criteria = list(sort_criteria)
    if match_criterion.name is None:
        return sort_criteria
    sort_criteria.append(
        mk.gui.Sort.create_criterion(
            match_criterion.name, ascending=False, source="match"
        )
    )
    return sort_criteria


sort_criteria = mk.Store([])
sort_criteria = add_match_criterion_to_sort(match.criterion, sort_criteria)
sort = mk.gui.Sort(df, criteria=sort_criteria)
df_sorted = sort(df)

gallery = html.div(
    mk.gui.Gallery(df_sorted, main_column=IMAGE_COLUMN),
    classes="h-full",
)


@mk.reactive()
def get_options(df):
    return [
        c
        for c in df.columns
        if "match" in c or c == "label" or c == "pred" or c == "correct"
    ]


select_x = mk.gui.core.Select(values=get_options(df), value="label")
select_y = mk.gui.core.Select(values=get_options(df), value="pred")
select_hue = mk.gui.core.Select(values=get_options(df), value="correct")
select_container = html.div(
    [
        html.div("x", classes="self-center"),
        select_x,
        html.div("y", classes="self-center"),
        select_y,
        html.div("hue", classes="self-center"),
        select_hue,
    ],
    classes="flex flex-row justify-center gap-4",
)


@mk.endpoint()
def set_filter_with_plot_selection(criteria: mk.Store, selected: List[str]):
    # Find any existing criterion for the primary key.
    pkey_filter_exists = False

    store = criteria
    # `criterion` can sometimes turn into a `dict` from a `FilterCriterion`
    # object. This is a bug that we need to fix.
    criteria = [
        FilterCriterion(**criterion) if isinstance(criterion, dict) else criterion
        for criterion in criteria
    ]
    for criterion in criteria:
        if criterion.column == "pkey":
            criterion.value = selected
            criterion.is_enabled = True
            pkey_filter_exists = True
            store.set(criteria)
            break
    if not pkey_filter_exists and selected:
        criteria.append(
            FilterCriterion(
                is_enabled=True,
                column="pkey",
                op="in",
                value=selected,
            )
        )
        store.set(criteria)


plot = mk.gui.plotly.ScatterPlot(
    df,
    x=select_x.value,
    y=select_y.value,
    hue=select_hue.value,
    title="Scatter Plot",
    on_select=set_filter_with_plot_selection.partial(criteria=filter.criteria),
)


component = html.div(
    [
        html.flexcol([match, filter, sort, select_container, plot], classes="h-full"),
        gallery,
    ],
    # Make a grid with two equal sized columns.
    classes="h-screen grid grid-cols-2 gap-4",
)

page = mk.gui.Page(component, id="error-analysis", progress=False)
page.launch()
