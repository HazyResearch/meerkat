# Tutorial: Filter & Sort

<!-- TODO: include screenshots? -->

In this tutorial, we'll explore some of the powerful built-in components of Meerkat. We'll start with our simple image gallery from [Tutorial 1](./tutorial-0.md), and then we'll add `Filter` and `Sort` components to allow users to filter and sort the images.

Through this tutorial, you will learn about:

- the `Filter` and `Sort` components
- `mk.gui.Store` objects
- composing components together to create more complex layouts

Below is code from [Tutorial 1](./tutorial-0.md) that we'll use as a starting point. It loads the `"imagenette"` dataset and creates a `Gallery` component to display the images.

```python
import meerkat as mk

df = mk.get("imagenette", version="160px")

gallery = mk.gui.Gallery(df=df, main_column="img", tag_columns=["label"])
```

## 📋 Filter

TODO: explanation

```python
filter_criteria = mk.gui.Store([])
filter = mk.gui.Filter(df=df, criteria=filter_criteria)
df = filter(df)
```

## 📋 Sort

TODO: explanation

```python
sort_criteria = mk.gui.Store([])
sort = mk.gui.Sort(df=df, criteria=sort_criteria)
df = sort(df)
```

## 🤲 Putting it all together

```python
import meerkat as mk

df = mk.get("imagenette", version="160px")

gallery = mk.gui.Gallery(df=df, main_column="img", tag_columns=["label"])

filter_criteria = mk.gui.Store([])
filter = mk.gui.Filter(df=df, criteria=filter_criteria)
df = filter(df)

sort_criteria = mk.gui.Store([])
sort = mk.gui.Sort(df=df, criteria=sort_criteria)
df = sort(df)

mk.gui.start()
page = mk.gui.Page(
    component=mk.gui.html.grid(
        slots=[filter, sort, gallery],
    ),
    id="filter-sort",
)
page.launch()
```

## 🎉 Conclusion

In just a few lines of code, we have a working app that allows users to filter and sort the image gallery. This is thanks to the powerful built-in components of Meerkat. You can browse the complete list of components [here](). In a later tutorial, you'll also learn how to make your own custom components that can be just as powerful.