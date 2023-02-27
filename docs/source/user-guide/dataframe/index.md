---
file_format: mystnb
kernelspec:
  name: python3
---

# DataFrame

<!-- Meerkat provides two data structures, the Column and the DataFrame, that together help you build, manage, and explore machine learning datasets. Everything you do with Meerkat will involve one or both of these data structures, so we begin this user guide with their high-level introduction. -->

A {class}`~meerkat.DataFrame` is a collection of equal-length columns (analagous to a [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame) in Pandas or R). DataFrames in Meerkat are used to manage datasets and per-example artifacts (*e.g.* model predictions and embeddings).  

Below we combine the columns we created above into a single DataFrame. We also add an additional column containing labels for the images. Note that we can pass non-Meerkat data structures like ``list``, ``np.ndarray``, ``pd.Series``, and ``torch.Tensor``  directly to the DataFrame constructor and Meerkat will infer the column type. We do not need to first convert to a Meerkat column.

```{code-cell} ipython3
df = mk.DataFrame(
    {
        "img": img_col,
        "label": ["boombox", "truck", "dog"],
        "id": id_col, 
    }
)
df
```

Read on to learn how we access the data in DataFrames.