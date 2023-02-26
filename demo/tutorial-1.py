"""Write a question answering interface."""
import os

from manifest import Manifest

import meerkat as mk
from meerkat.interactive.startup import file_find_replace

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
import os

from fastapi.staticfiles import StaticFiles

# print(os.getcwd())
# print(os.path.abspath("./meerkat/interactive/app/build/"))
# libpath = os.path.abspath("./meerkat/interactive/app/")
# api_url = "http://localhost:5000"
# file_find_replace(
#     libpath + "build",
#     r"(VITE_API_URL\|\|\".*?\")",
#     f'VITE_API_URL||"{api_url}"',
#     "*.js",
# )
# file_find_replace(
#     libpath + ".svelte-kit/output/client/_app/",
#     r"(VITE_API_URL\|\|\".*?\")",
#     f'VITE_API_URL||"{api_url}"',
#     "*.js",
# )
page().mount(
    "/",
    StaticFiles(
        directory=os.path.abspath("./meerkat/interactive/app/build/"), html=True
    ),
    "test",
)
# page().mount("/static", StaticFiles(directory=os.path.abspath("./temp/"),html = True), "test")
page.launch()
