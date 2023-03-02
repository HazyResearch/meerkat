import meerkat as mk
from meerkat.interactive.formatter.raw_html import HTMLFormatterGroup

df = mk.get("olivierdehaene/xkcd", registry="huggingface")["train"]

for col in df.columns:
    df[col] = df[col].to_numpy()

df["webpage"] = mk.files(df["url"]).format(HTMLFormatterGroup().defer())

filter = mk.gui.Filter(df)
filtered_df = filter(df)


gallery = mk.gui.html.div(
    [
        filter,
        mk.gui.Gallery(
            filtered_df.format(
                {
                    "explained_url": HTMLFormatterGroup(),
                }
            ),
            main_column="image_url",
        ),
    ],
    classes="flex flex-col h-[80vh]",
)


page = mk.gui.Page(component=gallery, id="xkcd")
page.launch()
