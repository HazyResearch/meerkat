# Tutorial 1: Image Gallery

In this tutorial, we will build a simple image gallery.

Through this tutorial, you will learn about:
- loading a dataset
- `mk.gui.start()`
- creating components
- `page.launch()`

The first step is to import `meerkat`! This will give you access to the meerkat DataFrame as well as the meerkat frontend components that will allow us to create our layouts.
```{code-cell} ipython3
:tags: [output_scroll]

import meerkat as mk
```

Next, we'll load in a dataset. Meerkat provides an interface into many datasets out of the box, but you are always free to download and use any dataset you like, or even make your own dataset. For this demo, we'll use `imagenette`, a subset of 10 easily classified classes from the Imagenet dataset. This dataset was created to be a lightweight version of Imagenet, useful for quickly trying out new ideas. In this line of code, meerkat will download the images and save them to a special directory: `~/.meerkat/datasets/imagenette`. This will make the data loading naerly instantaneous from here on out.
```{code-cell} ipython3
:tags: [output_scroll]

df = mk.get("imagenette")
```

The meerkat frontend is hosted on a port, so we'll have to spin up an instance, specifying a port for this to run on.
```{code-cell} ipython3
:tags: [output_scroll]

mk.gui.start(api_port=3000)
```

Let's create our frontend components now. Meerkat has numerous components for use out of the box, so we'll use the `Gallery` component to display our `df` DataFrame.
```{code-cell} ipython3
:tags: [output_scroll]

gallery = mk.gui.Gallery(df=df, main_column="img")
```

For the final step, we'll stick our `gallery` into a `Page` component, specifying an `id` of `"page"`, and we'll call `launch()` on our `page`. You'll see the layout render below!
```{code-cell} ipython3
:tags: [output_scroll]

page = mk.gui.Page(component=gallery, id="page")
page.launch()
```

Here is all six lines of code put together:
```{code-cell} ipython3
:tags: [output_scroll]

import meerkat as mk

mk.gui.start(api_port=3000)

df = mk.get("imagenette")
gallery = mk.gui.Gallery(df=df, main_column="img")

page = mk.gui.Page(component=gallery, id="page")
page.launch()
```

We have seen that 