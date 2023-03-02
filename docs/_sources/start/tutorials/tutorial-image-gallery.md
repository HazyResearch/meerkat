---
file_format: mystnb
kernelspec:
  name: python3
---

# Tutorial 2: Basic Interactive Apps (Building an Image Gallery)

<!-- TODO: include screenshots? -->

In this tutorial, we'll create our first Meerkat GUI by building an image gallery.

Through this tutorial, you will learn about:

- downloading a dataset
- starting a Meerkat interactive server
- building a layout with **components**
- launching an **app**

If you want to follow along, you can run the tutorial demo script.

```{code-block} bash
mk demo tutorial-image-gallery
```

As always, the first step is to import `meerkat`! This will give you access to the Meerkat `DataFrame` object as well as the Meerkat frontend components that will serve as the building blocks for our layouts.

```python
import meerkat as mk
```

## 💾 Downloading the dataset

Meerkat provides an interface into many datasets out of the box, but you are always free to download and use any dataset you like, or even make your own dataset.

For this demo, we're going to use the [Imagenette dataset](https://github.com/fastai/imagenette#image%E7%BD%91), a small subset of the original [ImageNet](https://www.image-net.org/update-mar-11-2021.php). This dataset is made up of 10 classes (e.g., "garbage truck", "gas pump", "golf ball"). This particular dataset offers three versions, so we'll choose the smallest by specifying `version="160px"`.

- Download time: <1 minute
- Download size: 206MB

Loading in this dataset will be lightning fast in the future, since Meerkat will cache the data locally.

```python
df = mk.get("imagenette", version="160px")
```

## 🕸️ Starting a Meerkat interactive server

Now that we have our dataset, we can spin up a Meerkat interactive server. This will allow us to view our dataset in the browser. By default, the interactive server will run on port 5000, but you can specify a different port if you'd like.

```python
mk.gui.start(api_port=3000)
```

## 🖼️ Building the layout

Let's create our frontend components. Meerkat has numerous components for use out of the box, so we'll use the `Gallery` component to display our `df` DataFrame. You can browse the complete list of components [here](). We'll specify the name of the column in our `df` where our images are stored, and we can optionally specify the names of a few columns to render as tags below each image.

```python
gallery = mk.gui.Gallery(df=df, main_column="img", tag_columns=["path", "label"])
```

For the final step, we'll stick our `gallery` into a `Page` component, specifying an `id` of `"gallery"`. Every Meerkat app must have a `Page` component as its root. The `id` is used to uniquely identify the page, and the `Page` is the only comment with the special `launch()` method.

```python
page = mk.gui.Page(component=gallery, id="gallery")
page.launch()
```

## 🚀 Launching the app

Here is all six lines of code put together:

```python
import meerkat as mk


df = mk.get("imagenette", version="160px")
mk.gui.start(api_port=3000)

gallery = mk.gui.Gallery(df=df, main_column="img", tag_columns=["path", "label"])

page = mk.gui.Page(component=gallery, id="gallery")
page.launch()
```

That's it! You can run this code either in a standalone `.py` file or directly in a Jupyter notebook. If running a `.py` file, use the command `mk run <file>.py --dev` and follow the link in the terminal to view your app in the browser. Any changes you make to the code will cause the page to automatically refresh. If running in a Jupyter notebook, the app should show up in the cell's output.

🥳 Congratulations on building your first Meerkat GUI.
