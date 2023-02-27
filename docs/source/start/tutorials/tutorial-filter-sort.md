# Tutorial: Filter & Sort

<!-- TODO: include screenshots? -->

In this tutorial, we'll explore some of the powerful built-in components of Meerkat. We'll start with our simple image gallery from [Tutorial 1](./tutorial-0.md), and then we'll add `Filter` and `Sort` components to allow users to filter and sort the images.

Through this tutorial, you will learn about:

- the `Filter` and `Sort` components
- `mk.Store` objects
- composing components together to create more complex layouts

As in [Tutorial 1](./tutorial-0.md), we'll use the [Imagenette dataset](https://github.com/fastai/imagenette#image%E7%BD%91). However, before creating a `Gallery` component, we'll need to first define our `Filter` and `Sort` components.

```python
import meerkat as mk

df = mk.get("imagenette", version="160px")
```

## ⛙ Combining components

First, we'll create a `Filter` component, passing in our original `df`, which will tell the component what to filter on. The returned object is a `mk.gui.Component`, which can be rendered on our page. At the same time, it is a callable function that can be invoked on our `df` to filter it. We'll see this later when creating the `Gallery`.

<!-- TODO: is this true? This is a common pattern in Meerkat, where components are both callable functions and `mk.gui.Component` objects. -->

```python
filter = mk.gui.Filter(df=df)
```

The `Sort` component works in nearly the identical way.

```python
sort = mk.gui.Sort(df=df)
```

Lastly, we can create a `Gallery` component, but instead of directly passing in the `df` object, we'll first call the `sort` and `filter` components on it. These functions will automatically be triggered anytime the filters or sorts are changed, and our `Gallery` will be updated accordingly in a reactive manner.

```python
gallery = mk.gui.Gallery(df=sort(filter(df)), main_column="img", tag_columns=["label"])
```

## 🤲 Putting it all together

To render our components, we will stick all three into a `Page` using the `mk.gui.html.grid` component. The complete code is shown below.

```python
import meerkat as mk

df = mk.get("imagenette", version="160px")

filter = mk.gui.Filter(df=df)
sort = mk.gui.Sort(df=df)
gallery = mk.gui.Gallery(df=sort(filter(df)), main_column="img", tag_columns=["label"])

mk.gui.start()
page = mk.gui.Page(
    component=mk.gui.html.grid(
        slots=[filter, sort, gallery],
    ),
    id="filter-sort",
)
page.launch()
```

We can run this app with the following command (or in Jupyter Notebook, simply run the code cell):

```bash
mk run filter-sort.py --dev
```

## 🎉 Conclusion

In just a few lines of code, we have a working app that allows users to filter and sort the image gallery. This is thanks to the powerful built-in components of Meerkat. You can browse the complete list of components {ref}`here <components_inbuilts>`. In a later tutorial, you'll also learn how to make your own custom components that can be just as powerful.
