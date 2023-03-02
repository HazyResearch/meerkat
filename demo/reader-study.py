"""A head-to-head reader study.

The reader inspects all variants of an image and selects the best one.
"""
import numpy as np
from PIL import Image

import meerkat as mk
from meerkat.interactive import html


def add_noise(img: Image) -> Image:
    """Simulate a noisy image."""
    img = np.asarray(img).copy()
    img += np.round(np.random.randn(*img.shape) * 10).astype(np.uint8)
    img = np.clip(img, 0, 255)
    return Image.fromarray(img)


@mk.endpoint()
def on_label(index, df: mk.DataFrame):
    """Add a label to the dataframe."""
    df["label"][row] = index
    row.set(row + 1)


@mk.endpoint()
def go_back(row: mk.Store, df: mk.DataFrame):
    row.set(max(0, row - 1))


@mk.endpoint()
def go_forward(row: mk.Store, df: mk.DataFrame):
    row.set(min(row + 1, len(df) - 1))


@mk.reactive()
def get_selected(label, value_list):
    # Label needs to be converted to a string because the values
    # are auto converted to strings by the RadioGroup component.
    # TODO: Fix this so that value list remains a list of ints.
    label = str(label)
    selected = value_list.index(label) if label in value_list else None
    return selected


@mk.reactive()
def select_row(df, row):
    # We have to do this because range indexing doesn't work with
    # stores.
    return df[row : row + 1]


df = mk.get("imagenette", version="160px")
df["noisy_img"] = mk.defer(df["img"], add_noise)
img_columns = ["img", "noisy_img"]
df["noisy_img"].formatters = df["img"].formatters

# Shuffle the dataset.
df = df.shuffle(seed=20)

# Randomize which images are shown on each gallery pane.
state = np.random.RandomState(20)
df["index"] = np.asarray(
    [state.permutation(np.asarray(img_columns)) for _ in range(len(df))]
)
df["img1"] = mk.defer(df, lambda df: df[df["index"][0]])
df["img2"] = mk.defer(df, lambda df: df[df["index"][1]])
df["img1"].formatters = df["img"].formatters
df["img2"].formatters = df["img"].formatters
anonymized_img_columns = ["img1", "img2"]

# Initialize labels to empty strings.
df["label"] = np.full(len(df), "")

df = df.mark()
row = mk.Store(0)
label = df[row]["label"]  # figure out why ["label"]["row"] doesn't work
cell_size = mk.Store(24)

back = mk.gui.core.Button(
    title="<",
    on_click=go_back.partial(row=row, df=df),
    classes="bg-slate-100 py-3 px-6 rounded-lg drop-shadow-md w-fit hover:bg-slate-200",  # noqa: E501
)
forward = mk.gui.core.Button(
    title=">",
    on_click=go_forward.partial(row=row, df=df),
    classes="bg-slate-100 py-3 px-6 rounded-lg drop-shadow-md w-fit hover:bg-slate-200",  # noqa: E501
)
label_buttons = [
    mk.gui.core.Button(
        title=f"{label}",
        on_click=on_label.partial(index=label, df=df),
        classes="bg-slate-100 py-3 px-6 rounded-lg drop-shadow-md w-fit hover:bg-slate-200",  # noqa: E501
    )
    for label in anonymized_img_columns
]
# We need to explicitly add the markdown hashes.
label_display = mk.gui.core.markdown.Markdown(body="## Label: " + label)

display_df = select_row(df, row)
galleries = [
    mk.gui.core.Gallery(display_df, main_column=main_column, cell_size=cell_size)
    for main_column in anonymized_img_columns
]


page = mk.gui.Page(
    html.flexcol(
        [
            html.gridcols2(galleries, classes="gap-4"),
            html.div(label_display),
            html.gridcols3(
                [
                    back,
                    html.gridcols2(label_buttons, classes="gap-x-2"),
                    forward,
                ],
                classes="gap-4 self-center justify-self-center w-full items-center",
            ),
        ]
    ),
    id="reader-study",
)
page.launch()
