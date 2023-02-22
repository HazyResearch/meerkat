---
file_format: mystnb
kernelspec:
  name: python3
---

# Introduction to DataFrames 

Meerkat provides two data structures, the column and the dataframe, that together help 
you build, manage, and explore machine learning datasets . Everything you do with Meerkat will 
involve one or both of these data structures, so we begin this user guide with their
high-level introduction. 

## Column

A column is a sequential data structure (analagous to a [`Series`](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) in Pandas or a [`Vector`](https://cran.r-project.org/doc/manuals/r-release/R-intro.html#Simple-manipulations-numbers-and-vectors) in R). 
Meerkat supports a diverse set of column types (*e.g.* {py:class}`TensorColumn <meerkat.TensorColumn>`, 
{py:class}`ImageColumn <meerkat.ImageColumn>`), each intended for different kinds of data. To see a
list of the core column types and their capabilities, see {py:doc}`column_types <column_types>`.

Below we create a simple column to hold a set of images stored on disk. To create it,
we simply pass filepaths to the {py:class}`ImageColumn <meerkat.ImageColumn>` constructor.

```{code-cell} ipython3
:tags: [remove-cell]
import os
import meerkat as mk
abs_path_to_img_dir = os.path.join(os.path.dirname(os.path.dirname(mk.__file__)), "docs/assets/guide/data_structures")
```

```{code-cell} ipython3
img_col = mk.image(
    ["img_0.jpg", "img_1.jpg", "img_2.jpg"], 
    base_dir=abs_path_to_img_dir
)
img_col
```

All Meerkat columns are subclasses of {py:class}`Column <meerkat.Column>` and share a common 
interface, which includes 
{py:meth}`__len__ <meerkat.Column.__len__>`,
{py:meth}`__getitem__ <meerkat.Column.__getitem__>`, 
{py:meth}`__setitem__ <meerkat.Column.__setitem__>`, 
{py:meth}`filter <meerkat.Column.filter>`, 
{py:meth}`map <meerkat.Column.map>`, 
and {py:meth}`concat <meerkat.Column.concat>`. Below we get the length of the column we just created. 

```{code-cell} ipython3
len(img_col)
```


Certain column types may expose additional functionality. For example, 
{py:class}`TensorColumn <meerkat.TensorColumn>`
inherits most of the functionality of an
[`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html).

```{code-cell} ipython3
id_col = mk.TensorColumn([0, 1, 2])
id_col.sum()
id_col == 1
```

If you don't know which column type to use, you can just pass a familiar data 
structure like a ``list``, ``np.ndarray``, ``pd.Series``, and ``torch.Tensor`` to 
{py:meth}`Column.from_data <meerkat.Column.from_data>`
and Meerkat will automatically pick an appropriate column type. 

```{code-cell} ipython3
import torch
tensor = torch.tensor([1,2,3])
mk.Column.from_data(tensor)
```

## DataFrame

A 
{py:class}`DataFrame <meerkat.DataFrame>`
is a collection of equal-length columns (analagous to a 
[`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame) in Pandas or R). 
DataFrames in Meerkat are used to manage datasets and per-example artifacts (*e.g.* model predictions and embeddings).  

Below we combine the columns we created above into a single DataFrame. We also add an 
additional column containing labels for the images. Note that we can pass non-Meerkat data 
structures like ``list``, ``np.ndarray``, ``pd.Series``, and ``torch.Tensor``  directly to the 
DataFrame constructor and Meerkat will infer the column type. We do not need to first 
convert to a Meerkat column. 

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

Read on to learn how we access the data in Columns and DataFrames.