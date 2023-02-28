# A Simple DataFrame Viewer

Let's run through a simple example of how to build a DataFrame viewer, combining Meerkat's `DataFrame` with interactivity. We'll build a simple application that allows users to filter the `imagenette` dataset by label. We'll display the filtered DataFrame in a `Table` view.

## Import

As always, we first start by importing Meerkat.

```python
import meerkat as mk
```

In addition to being a library for building interactive applications, Meerkat is also a library for working with data. It provides a `DataFrame` object that is similar to Pandas, but is designed to work with unstructured data.

This deep integration between the `DataFrame` and the interactive GUIs makes it easy to build applications that work with data, and we're excited to see what users build with it!

## Load DataFrame

We'll start by loading a small image dataset called `imagenette` from Meerkat's dataset registry. It's a great image classification dataset for demos!

```python
# Let's load the `imagenette` dataset using Meerkat's dataset registry
df: mk.DataFrame = mk.get("imagenette")
```

## Choice Component

Next, we'll create a `Choice` component that allows users to select a label to filter by.

```python
# List of unique labels represented in the DataFrame
labels = list(df['label'].unique())

# Component that can display the labels as choices to select from
choices = mk.gui.Choice(choices=labels, value=labels[0], title="Choose a Label")
```

Note here that `Choice.value` is a `Store`. This means that when the user selects a new label in the GUI, the `value` will be updated automatically!

## Reactive Functions

Now, we'll see another concept that Meerkat's interactive GUIs are built on: `reactive functions`. These are Python functions that are automatically re-run when the inputs change. They make it easy to build reactive applications that are driven by data.

```python
# Create a reactive function that returns a filtered view of the DataFrame
@mk.reactive()
def filter_by_label(df: mk.DataFrame, label: str):
    """If either the `df` or `label` argument changes, this function will be re-run."""
    return df[df['label'] == label]

# Run the reactive function in the `react` context manager to make it retrigger automatically
with mk.reactive():
    # Reactively run filter by label with `df` and `choices.value`
    df_filtered = filter_by_label(df, choices.value)
```

By wiring up the `choices.value` to the `label` argument, we can automatically filter the DataFrame when the user selects a new label.

## Table Component

Now, we can target the filtered DataFrame with a `Table` component to display it.

```python
# Visualize the filtered_df
df_viz = mk.gui.Table(df=df_filtered)
```

## Interface

And then finally, we can launch the interface!

```python
mk.gui.start()
mk.gui.Interface(
    component=mk.gui.RowLayout(components=[choices, df_viz])
).launch()
```
