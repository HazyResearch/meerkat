import meerkat as mk

df = mk.DataFrame(
    {
        "img_url": [
            "https://placeimg.com/200/200/animals",
            "https://placeimg.com/200/200/people",
            "https://placeimg.com/200/200/tech",
            "https://placeimg.com/200/200/nature",
            "https://placeimg.com/200/200/architecture",
        ]
    }
)
df["img"] = mk.image(df["img_url"])

gallery = mk.gui.Gallery(df=df, main_column="img", tag_columns=["img_url"])

interface = mk.gui.Interface(component=gallery, id="image")
interface.launch()
