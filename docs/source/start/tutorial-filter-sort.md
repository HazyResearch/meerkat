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

This only requires three lines of code, but each one has a very specific purpose. The first line creates a `Store` object, which is a special Meerkat object that wraps any type of data to make it accessible to reactive components. The second line creates a `Filter` component, which takes this `filter_criteria` as a parameter. The third line applies the filter to the `df` DataFrame reactively, so that whenever the `filter_criteria` changes, the `df` will be updated. The `filter` function is reactive because it is a `mk.gui.Component`, and all Meerkat components are reactive by definition.

TODO: make explanation better
```python
filter_criteria = mk.gui.Store([])
filter = mk.gui.Filter(df=df, criteria=filter_criteria)
df = filter(df)
```

## 📋 Sort

The `Sort` component works in nearly the identical way. Note that we have to create an additional `Store` object that will be tied to the `Sort` component. Again, the `sort` function is reactive because it is a `mk.gui.Component`, meaning it will automatically be invoked whenever there is a change to `sort_criteria`.

TODO: make explanation better
```python
sort_criteria = mk.gui.Store([])
sort = mk.gui.Sort(df=df, criteria=sort_criteria)
df = sort(df)
```

## 🤲 Putting it all together

To combine all our components, we will stick them into a `Page` using the `mk.gui.html.grid` component. The complete code is shown below.

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