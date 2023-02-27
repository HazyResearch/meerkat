import meerkat as mk
from meerkat.interactive.app.src.lib.shared.cell.website import WebsiteFormatter
from meerkat.interactive.formatter.image import ImageFormatter

df = mk.get("olivierdehaene/xkcd", registry="huggingface")["train"]

for col in df.columns:
    df[col] = df[col].to_numpy()

filter = mk.gui.Filter(df)
filtered_df = filter(df)

gallery = mk.gui.html.div(
    [
        filter,
        mk.gui.Gallery(
            filtered_df.format(
                {
                    "image_url": ImageFormatter(max_size=(300, 300)),
                    "url": WebsiteFormatter(height=30),
                    "explained_url": WebsiteFormatter(height=30),
                }
            ),
            main_column="image_url",
        ),
    ],
    classes="flex flex-col h-[80vh]",
)


page = mk.gui.Page(component=gallery, id="xkcd")
page.launch()
