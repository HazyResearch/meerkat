

import meerkat as mk
from meerkat.interactive.app.src.lib.component.image_url import ImageUrl


col = mk.column([
    "http://commons.wikimedia.org/wiki/Special:FilePath/Mary%2C%20Untier%20of%20Knots%20by%20Schmidtner.png?width=100&thumb=true",
    "http://commons.wikimedia.org/wiki/Special:FilePath/Amor%20als%20Sieger%20-%20Gem%C3%A4ldegalerie%20Berlin%20-%205139072.jpg?width=100&thumb=true"
])

df = mk.DataFrame(
    {
        "img_url": col,
    }
)
image = ImageUrl(
    url="https://placeimg.com/200/200/animals"
)


grayscale = mk.gui.Store(False)
button = mk.gui.Toggle(value=grayscale)
df["img_url"] = df["img_url"].format(ImageUrl.to_formatter(grayscale=grayscale))

gallery = mk.gui.Gallery(df=df, main_column="img_url")
interface = mk.gui.Interface(component=mk.gui.RowLayout(slots=[button, gallery]), id="image")
interface.launch() 


# import meerkat as mk
# from meerkat.interactive.app.src.lib.component.image import Image


# col = mk.column([
#     "http://commons.wikimedia.org/wiki/Special:FilePath/Mary%2C%20Untier%20of%20Knots%20by%20Schmidtner.png?width=100&thumb=true",
#     "http://commons.wikimedia.org/wiki/Special:FilePath/Amor%20als%20Sieger%20-%20Gem%C3%A4ldegalerie%20Berlin%20-%205139072.jpg?width=100&thumb=true"
# ])

# df = mk.DataFrame( 
#     {
#         "img_url": col,
#         "img": mk.ImageColumn(col)
#     }
# )




# gallery = mk.gui.Gallery(df, main_column="img")

# image = mk.gui.Image("http://commons.wikimedia.org/wiki/Special:FilePath/Mary%2C%20Untier%20of%20Knots%20by%20Schmidtner.png?width=100&thumb=true")

# button = mk.gui.Button("Click me!")

# formatter = mk.gui.Image.to_formatter(
#     grayscale=button.value
# )
# for 
#     col.formatter = formatter


# encoder = encode.partial(button.value)

# df = df["col1", "col2"]

# for col in ["col1", "col2"]:
#     store = buttons[i].value
#     col.formatter.grayscale = store


# store = df["col1"].formatter.grayscale
# button1.value = store
# button2.value = 


# col.format(
#     mk.gui.Image.to_formatter(**kwargs),
#     size="small"
# )

# class Formatter:
#     component: mk.gui.Image

#     def encode():
#         pass 

# mk.gui.Gallery


# df["img"] = df["img"].format(Image(grayscale=True))

