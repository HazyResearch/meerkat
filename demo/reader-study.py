import numpy as np
from PIL import Image

import meerkat as mk


def add_noise(img: Image) -> Image:
    """Add noise to the image."""
    img = np.asarray(img).copy()
    img += np.round(np.random.randn(*img.shape) * 10).astype(np.uint8)
    img = np.clip(img, 0, 255)
    return Image.fromarray(img)


@mk.gui.endpoint
def on_label(index):
    # Add the label to the dataframe.
    # Every time the user clicks on a label, we should add it to the dataframe.
    df["label"][row] = index
    row.set(row + 1)
    print("Dataframe", df["label"][: row.value].data)


@mk.gui.endpoint
def go_back(row):
    row.set(max(0, row - 1))


@mk.gui.endpoint
def go_forward(row, df):
    row.set(min(row + 1, len(df) - 1))


@mk.gui.react()
def get_selected(label, value_list):
    # Label needs to be converted to a string because the values
    # are auto converted to strings by the RadioGroup component.
    # TODO: Fix this so that value list remains a list of ints.s
    label = str(label)
    selected = value_list.index(label) if label in value_list else None
    return selected


@mk.gui.react()
def select_row(df, row):
    # We have to do this because range indexing doesn't work with
    # stores.
    return df[row : row + 1]


df = mk.get("imagenette", version="160px")
df["noisy_img"] = mk.defer(df["img"], add_noise)
# Shuffle the dataset.
df = df.shuffle(seed=20)
# We want to randomize which images are shown on each gallery pane.
# TODO: Add a column indicating which column should be displayed.
df["img1"] = df["img"]
df["img2"] = df["noisy_img"]
# Initialize labels to -1.
df["label"] = -np.ones(len(df), dtype=np.int32)

with mk.gui.react():
    row = mk.gui.Store(0)
    label = df[row]["label"]  # figure out why ["label"]["row"] doesn't work

    values = mk.gui.Store([0, 1])
    selected = get_selected(label, values)
    print("Selected inode", selected.inode.id)
    radio = mk.gui.core.RadioGroup(
        name="Better Image",
        values=values,
        selected=selected,
        on_change=on_label,
        classes="bg-violet-50 p-2 rounded-lg w-fit flex items-center justify-center",
    )
    back = mk.gui.core.Button(
        title="<",
        on_click=go_back.partial(row),
        classes="bg-slate-100 py-3 px-6 rounded-lg drop-shadow-md w-fit hover:bg-slate-200",  # noqa: E501
    )
    forward = mk.gui.core.Button(
        title=">",
        on_click=go_forward.partial(row=row, df=df),
        classes="bg-slate-100 py-3 px-6 rounded-lg drop-shadow-md w-fit hover:bg-slate-200",  # noqa: E501
    )

    display_df = select_row(df, row)
    img1 = mk.gui.core.Gallery(df=display_df, main_column="img1")
    img2 = mk.gui.core.Gallery(df=display_df, main_column="img2")

mk.gui.start(shareable=False)
page = mk.gui.Page(
    component=mk.gui.html.flexcol(
        [
            mk.gui.html.gridcols2([img1, img2], classes="gap-4"),
            mk.gui.html.gridcols3([back, radio, forward], classes="gap-4"),
        ],
        classes="gap-y-4",
    ),
    id="reader-study",
)
page.launch()
