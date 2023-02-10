import meerkat as mk
from meerkat.interactive.app.src.lib.component.core.image import DeferredImageFormatter, ImageFormatter


df = mk.get("olivierdehaene/xkcd", registry="huggingface")['train']
for col in df.columns:
    df[col] = df[col].to_numpy()

df['image_url'].formatter = ImageFormatter()

filter = mk.gui.Filter(df=df)

with mk.gui.react():
    filtered_df = filter(df)

# Make the gallery occupy only 50% of the screen height
gallery = mk.gui.html.div(
    [
        filter,
        mk.gui.Gallery(filtered_df, main_column="image_url"),
    ],
    classes="flex flex-col h-[80vh]",
)



page = mk.gui.Page(component=gallery, id="xkcd")
page.launch()
