---
file_format: mystnb
kernelspec:
  name: python3
---

(guide/interactive/formatters/overview)=
# Formatters
Meerkat GUIs can display data of many different types, from images to text to audio.  Formatters control how these data types are displayed and interacted with in Meerkat GUIs.

For example, images can be displayed using the {class}`meerkat.format.ImageFormatter`. 
Each formatter, specifies optional parameters that can be used to configure how the data is displayed. For example, the {class}`meerkat.format.ImageFormatter` has a `max_size` parameter that can be used to specify the maximum size of the image to display.

  
```{code-cell} ipython3
import meerkat as mk
formatter = mk.format.ImageFormatter(max_size=(224, 224))
```

## Formatter Group
Data in a Meerkat column sometimes need to be displayed differently in different GUI contexts. For example, in a table, we display thumbnails of images, but in a carousel view, we display the full image. 

Because most components in Meerkat work on any data type, it is important that they are implemented in a formatter-agnostic way. So, instead of specifying formatters, components make requests for data specifying a *formatter placeholder*. For example, the {class}`mk.gui.Gallery` component requests data using the `thumbnail` formatter placeholder.

For a specific column of data, we specify which formatters to use for each placeholder using a *formatter group*. A formatter group is a mapping from formatter placeholders to formatters. Each column in Meerkat has a `formatter_group` property. A column's formatter group controls how it will be displayed in different contexts in Meerkat GUIs. 

Much of the time, you don't need to worry about specifying a formatter group, each column automatically populates its formatter group with sensible defaults. 

For example, let's take a look at the formatter group for a column of images in the `imagenette` dataset.
```{code-cell} ipython3
import meerkat as mk

df = mk.get("imagenette")
df["img"].formatter_group
```

This tells Meerkat components to use `ImageFormatter(max_size=(224, 224))` when displaying thumbnails of column values, as in the {class}`mk.gui.Table` component. 

## Changing Formatters 

There are two ways you can use formatters to control how data is displayed in Meerkat GUIs.
1. By updating a column's formatter:
```{code-cell} ipython3
df["img"].formatter_group["thumbnail"] = ImageFormatter(max_size=(48, 48))
```

2. By specifying it when creating a component:
```python
df.columns # ['img', 'text']
gallery = Gallery(
  formatters={
    "img": {"icon": ImageFormatter(max_size=(48, 48))},
    "path": {"icon": TextFormatter()},
  }
)
```


## Implementing a Formatter
You can implement your own formatter for a custom data type. 
A formatter implementation must specify three things: a `component_class`, an `encode` method, and a `props` method. 

Consider the following example of a formatter that encodes images as base64 strings and sends them to the frontend to be displayed using the `Image` component.

```python
class ImageFormatter(Formatter):
  component_class: Type[BaseComponent] = Image

  def __init__(self, max_size: Tuple[int]=None, classes: str, grayscale: str):
    self.max_size = max_size
    self.classes = classes
    self.grayscale = grayscale

  def encode(self, cell: PIL.Image) -> str:
    with BytesIO() as buffer:
      if max_size:
        image.thumbnail(max_size)
      image.save(buffer, "jpeg")
      return "data:image/jpeg;base64,{im_base_64}".format(
        im_base_64=base64.b64encode(buffer.getvalue()).decode()
      )
  
  @property
  def props(self) -> Dict[str, Any]:
    return {
      "classes": self.classes,
      "grayscale": self.grayscale
    }
```
Let's break down what's happening here. 
- `component_class` specifies the class of the frontend component that should be created to display the data. In this case, it is the `Image` component defined below.
```python
class Image(Component):
    data: str
    classes: str = ""
    grayscale: str = False
```
- `props` specifies the values of the properties passed when constructing the components. Notice that the keys in the returned dictionary match the names of the properties defined in the `Image` component above. 

- `encode` specifies how a single cell from a column should be encoded on the Python side before being sent up to the frontend. In this case we are encoding an image as a base-64 string.

## Formatter Placeholders
As we discussed above, components specify formatter placeholders when requesting data from a column. This allows them to be formatter-agnostic, while still being able to display data in different ways depending on the context.

Formatter placeholders have special names, such as `icon`, `focus`, and `thumbnail`, which
allow you to configure them when using Meerkat components in Python.

Components can use one or more of these placeholders (or define custom formatter placeholders) in order to change the encoding of the data fetched from the Python backend. They can pass formatter placeholders to the data fetching API provided by Meerkat, and data in the requested format is then delivered to them.

You can also define custom formatter placeholders that you want to associate with a frontend component that you might be building. For example, say you want to define a `anonymous` formatter that will be used to display data in an anonymized way e.g. to blur images, videos and text, and scramble audio.

```python
class Anonymous(FormatterPlaceholder):
  """
  A placeholder that represents formatters that anonymize data.
  
  Ideally, should be configured by formatters that blur or 
  deidentify data e.g. blur images, videos, etc.
  """
  fallbacks: [Small]
```

Here, you use `fallbacks` to specify a formatter variable to use instead of `anonymous` if the user forgets to configure it. All formatter placeholders fallback to using `base` automatically, which uses a standard encoding of all data types to make sure everything can be displayed.

You would then use this `anonymous` variable in your frontend code when fetching data using the Meerkat `js` API.

```js
// Fetch data from the backend using the anonymous formatter.
data = await fetchChunk({
  ...,
  formatter: "anonymous"
});
```

And you can implement custom formatters to achieve the effect of anonymizing data before sending it to the frontend e.g. using a blur filter over image data in the `encode` method of an `ImageBlurFormatter`.

When and where do we tell Meerkat how to associate formatter placeholders to these formatters? This is where formatter groups are used, which we discuss next.


As a user, when using a component with one or more formatter placeholders, you can pass in the formatter you want to use for each formatter variable. Meerkat will automatically ensure that the frontend gets data using the `encode` method associated with the formatter that was passed in.

