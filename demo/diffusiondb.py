from PIL import Image

import meerkat as mk

# Load the dataset
df = mk.get(
    "poloclub/diffusiondb",
    version="large_random_1k",
    registry="huggingface",
)["train"]
for col in df.columns:
    df[col] = df[col].to_numpy()

df["image_path"] = df["image"].defer(lambda x: x["path"])
df["image"] = (
    df["image_path"]
    .map(lambda x: Image.open(x))
    .format(mk.format.ImageFormatterGroup())
)


# Add a filtering component
df = df.mark()
filter = mk.gui.Filter(df)
df_filtered = filter(df)
df_grouped = df_filtered.groupby("cfg").count()
df_grouped = df_grouped.rename({"height": "count"})

# Visualize the images in a gallery
gallery = mk.gui.Gallery(df_filtered, main_column="image")

# Add a plot component
plot = mk.gui.plotly.BarPlot(df_grouped, x="cfg", y="count")

table = mk.gui.Table(df_grouped)

component = mk.gui.html.gridcols2(
    [
        mk.gui.html.flexcol(
            [
                mk.gui.core.Header("DiffusionDB"),
                filter,
                gallery,
            ],
        ),
        mk.gui.html.flexcol(
            [
                mk.gui.core.Header("Distribution"),
                plot,
                table,
            ],
            classes="justify-center",
        ),
    ],
    classes="gap-4",
)

# Make a page and launch
page = mk.gui.Page(component, id="diffusiondb")
page.launch()
