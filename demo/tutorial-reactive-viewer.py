"""A reactive image viewer that allows you to select a class and see 16 random
images from that class.

This is a tutorial on how to use `reactive` functions in Meerkat, to
build complex reactive workflows.
"""

import meerkat as mk

df = mk.get("imagenette", version="160px")
IMAGE_COL = "img"
LABEL_COL = "label"


@mk.reactive()
def random_images(df: mk.DataFrame):
    images = df.sample(16)[IMAGE_COL]
    formatter = images.formatters["base"]
    # formatter = images.formatters['tiny']
    return [formatter.encode(img) for img in images]


labels = list(df[LABEL_COL].unique())
class_selector = mk.gui.Select(
    values=list(labels),
    value=labels[0],
)

# Note that neither of these will work:
# filtered_df = df[df[LABEL_COL] == class_selector.value]
#       (doesn't react to changes in class_selector.value)
# filtered_df = mk.reactive(lambda df: df[df[LABEL_COL] == class_selector.value])(df)
#       (doesn't react to changes in class_selector.value)
filtered_df = mk.reactive(lambda df, label: df[df[LABEL_COL] == label])(
    df, class_selector.value
)

images = random_images(filtered_df)

# This won't work with a simple reactive fn like a random_images
# that only has df.sample
# as the encoding needs to be done in the reactive fn
# grid = mk.gui.html.gridcols2([
#   mk.gui.Image(data=images.formatters["base"].encode(img)) for img in images
# ])

grid = mk.gui.html.div(
    [
        # Make the image square
        mk.gui.html.div(mk.gui.Image(data=img))
        for img in images
    ],
    classes="h-fit grid grid-cols-4 gap-1",
)

layout = mk.gui.html.div(
    [
        mk.gui.html.div(
            [mk.gui.Caption("Choose a class:"), class_selector],
            classes="flex justify-center items-center mb-2 gap-2",
        ),
        grid,
    ],
    classes="h-full flex flex-col m-2",
)

page = mk.gui.Page(layout, id="reactive-viewer")
page.launch()
