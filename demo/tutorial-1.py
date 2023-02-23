import meerkat as mk

df = mk.get("imagenette", version="160px")
IMAGE_COL = "img"
LABEL_COL = "label"


# df.mark()


@mk.reactive()
def random_images(df: mk.DataFrame):
    images = df.sample(16)[IMAGE_COL]
    formatter = images.formatters['base']
    # formatter = images.formatters['tiny']
    return [formatter.encode(img) for img in images]


labels = list(df[LABEL_COL].unique())
class_selector = mk.gui.Select(
    values=list(labels),
    value=labels[0],
)

# This won't work!
# filtered_df = df[df[LABEL_COL] == class_selector.value]
# This won't work!
# filtered_df = mk.reactive(lambda df: df[df[LABEL_COL] == class_selector.value])(df)

filtered_df = mk.reactive(lambda df, label: df[df[LABEL_COL] == label])(df, class_selector.value)

images = random_images(filtered_df)

# This won't work with a simple reactive fn like a random_images that only has df.sample
# as the encoding needs to be done in the reactive fn
# grid = mk.gui.html.gridcols2([mk.gui.Image(data=images.formatters["base"].encode(img)) for img in images])

# Basic layout
# grid = mk.gui.html.gridcols2([mk.gui.Image(data=img) for img in images])

# Better layout
grid = mk.gui.html.gridcols4([
    # Make the image square
    mk.gui.html.div(mk.gui.Image(data=img), style="aspect-ratio: 1 / 1")
    for img in images
], classes="gap-2")


# layout = mk.gui.html.flexcol(
#     [
#         class_selector,
#         grid,
#     ]
# )

layout = mk.gui.html.flexcol(
    [
        mk.gui.html.div(
            [mk.gui.Caption("Choose a class:"), class_selector], 
            classes="flex justify-center items-center mb-2 gap-4"),
        grid,
    ]
)

page = mk.gui.Page(component=layout, id="tutorial-1")
page.launch()
