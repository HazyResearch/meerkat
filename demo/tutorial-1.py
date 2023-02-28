"""Write a question answering interface."""
import os

from fastapi.staticfiles import StaticFiles

import meerkat as mk

df = mk.get("imagenette", version="160px")
IMAGE_COL = "img"
LABEL_COL = "label"


@mk.reactive()
def random_images(df: mk.DataFrame):
    images = df.sample(16)[IMAGE_COL]
    formatter = images.formatters["base"]
    return [formatter.encode(img) for img in images]


labels = list(df[LABEL_COL].unique())
class_selector = mk.gui.Select(
    values=list(labels),
    value=labels[0],
)

filtered_df = mk.reactive(lambda df, label: df[df[LABEL_COL] == label])(
    df, class_selector.value
)

images = random_images(filtered_df)

grid = mk.gui.html.gridcols4(
    [
        mk.gui.html.div(mk.gui.Image(data=img), style="aspect-ratio: 1 / 1")
        for img in images
    ],
    classes="gap-2",
)


layout = mk.gui.html.flexcol(
    [
        mk.gui.html.div(
            [mk.gui.Caption("Choose a class:"), class_selector],
            classes="flex justify-center items-center mb-2 gap-4",
        ),
        grid,
    ]
)
page = mk.gui.Page(component=layout, id="tutorial-1")

page().mount(
    "/",
    StaticFiles(
        directory=os.path.abspath("./meerkat/interactive/app/build/"), html=True
    ),
    "test",
)
page.launch()
