from datetime import datetime
from typing import List

import meerkat as mk
from meerkat.interactive import html
from meerkat.interactive.app.src.lib.component.core.filter import FilterCriterion

# Data loading.
imagenette = mk.get("imagenette", version="160px")
imagenette = imagenette[["img_id", "path", "img", "label", "index"]]
df = mk.DataFrame.read(
    "https://huggingface.co/datasets/meerkat-ml/meerkat-dataframes/resolve/main/imagenette-160px-facebook-convnext-tiny-224.mk.tar.gz",  # noqa: E501
    overwrite=False,
)
# df = mk.DataFrame.read("~/Downloads/imagenette-remapped.mk")
df = imagenette.merge(df[["img_id", "logits", "pred"]], on="img_id")
df["logits"] = df["logits"].data
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


@mk.endpoint()
def add_match_criterion_to_sort(criterion, sort_criteria: mk.Store[List]):
    # make a copy of the sort criteria.
    criteria = sort_criteria
    if criterion.name is None:
        return
    sort_criteria = [
        criterion for criterion in sort_criteria if criterion.source != "match"
    ]
    sort_criteria.insert(
        0, mk.gui.Sort.create_criterion(criterion.name, ascending=False, source="match")
    )
    criteria.set(sort_criteria)


sort = mk.gui.Sort(df)
match = mk.gui.Match(
    df,
    against=EMBED_COLUMN,
    on_match=add_match_criterion_to_sort.partial(sort_criteria=sort.criteria),
)
filter = mk.gui.Filter(df)
df = filter(df)
df_sorted = sort(df)

gallery = mk.gui.Gallery(df_sorted, main_column=IMAGE_COLUMN)


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
        html.div("x", classes="self-center font-mono"),
        select_x,
        html.div("y", classes="self-center font-mono"),
        select_y,
        html.div("hue", classes="self-center font-mono"),
        select_hue,
    ],
    classes="w-full flex flex-row justify-center gap-4",
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

# ================ Notes =================


def capture_state():
    """Capture the state of the application at the time the."""
    return {
        "filter": filter.criteria.value,
        "sort": sort.criteria.value,
        "match": match.criterion.value,
        "select_x": select_x.value.value,
        "select_y": select_y.value.value,
        "select_hue": select_hue.value.value,
    }


def set_state(state=None):
    if state:
        filter.criteria.set(state["filter"])
        sort.criteria.set(state["sort"])
        match.criterion.set(state["match"])
        select_x.value.set(state["select_x"])
        select_y.value.set(state["select_y"])
        select_hue.value.set(state["select_hue"])

    out = (
        filter.criteria,
        sort.criteria,
        match.criterion,
        select_x.value,
        select_y.value,
        select_hue.value,
    )
    assert all(isinstance(o, mk.Store) for o in out)
    return out


@mk.endpoint()
def restore_state(notes: mk.DataFrame, selected: List[int]):
    print(selected, type(selected))
    if not selected:
        return

    if len(selected) > 1:
        raise ValueError("Can only select one row at a time.")

    state = notes.loc[selected[0]]["state"]
    set_state(state)


@mk.endpoint()
def add_note(df: mk.DataFrame, notepad_text: mk.Store[str], text):
    new_df = mk.DataFrame(
        {"time": [str(datetime.now())], "notes": [text], "state": [capture_state()]}
    )
    if len(df) > 0:
        new_df = new_df.append(df)

    # Clear the text box.
    notepad_text.set("")
    df.set(new_df)

    # Clear the selection
    # selected.set([])


notes = mk.DataFrame(
    {
        "time": [str(datetime.now())],
        "notes": ["An example note."],
        "state": [capture_state()],
    }
).mark()

notepad_text = mk.Store("")
selected = mk.Store([])
notepad = mk.gui.Textbox(
    text=notepad_text,
    placeholder="Add your observations...",
    classes="w-full h-10 px-3 rounded-md shadow-md my-1 border-gray-400",
    on_keyenter=add_note.partial(df=notes, notepad_text=notepad_text),
)
notes_table = mk.gui.Table(
    notes[["time", "notes"]],
    selected=selected,
    single_select=True,
    on_select=restore_state.partial(notes=notes),
    classes="h-full pad-x-5 pl-2",
)


# ================ Display =================

component = html.div(
    [
        html.div(
            [
                match,
                filter,
                sort,
                select_container,
                plot,
                html.div(
                    [
                        html.div(
                            "Notes",
                            classes="font-bold text-md text-slate-600 self-start pl-1",
                        ),
                        html.div(notepad, classes="px-1"),
                        notes_table,
                    ],
                    classes="bg-slate-100 px-1 py-2 gap-y-4 rounded-lg w-full h-fit",
                ),
            ],
            classes="grid grid-rows-[auto_auto_auto_auto_5fr_3fr] h-full",
        ),
        gallery,
    ],
    # Make a grid with two equal sized columns.
    classes="h-screen grid grid-cols-2 gap-4",
)

page = mk.gui.Page(component, id="error-analysis", progress=False)
page.launch()
