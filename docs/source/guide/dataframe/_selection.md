# Data Selection

As discussed in the {doc}`data_structures`, there are two key data structures in Meerkat: the Column and the DataFrame. In this guide, we'll demonstrate how to access the data stored within them.

Throughout, we'll be selecting data from the following DataFrame, which holds the Imagenette dataset, a small subset of the original ImageNet. This DataFrame includes a column holding images, a column holding their labels, and a few others.

```{eval-rst}
.. ipython:: python

   import meerkat as mk
   df = mk.datasets.get("imagenette")
   df

   @suppress
   from display import display_df
   @suppress
   display_df(df.head()[["img", "label", "label_id", "label_idx", "split", "img_path"]], "imagenette_head")





```

## Selecting Columns

The columns in a DataFrame are uniquely identified by `str` names. The code
below displays the column names in the Imagenette dataframe we loaded above:

```{eval-rst}
.. ipython:: python

   df.columns
```

Using these column names, we can pull out an individual column or a subset of them as a new
DataFrame.

```{eval-rst}
.. card::


    **Selecting a Single Column**: ``str`` -> :class:`~meerkat.Column`
    ^^^^^^^^^^^^^^

    To select a single column, we simply pass it's name to the index operator. For example,

    .. ipython:: python

        col = df["label"]
        col

    Passing a ``str`` that isn't among the column names will raise a ``KeyError``.
```

It may be helpful to think of a DataFrame as a dictionary mapping column names to columns.
Indeed, a DataFrame implements other parts of the `dict` interface including {meth}`~meerkat.DataFrame.keys()`, {meth}`~meerkat.DataFrame.values()`, and {meth}`~meerkat.DataFrame.items()`. Unlike a dictionary, multiple columns in a DataFrame can be selected at once.

```{eval-rst}
.. card::


    **Selecting Multiple Columns**: ``Sequence[str]`` -> :class:`~meerkat.DataFrame`
    ^^^^^^^^^^^^^^

    You can select multiple columns by passing a list or tuple of column names. Doing so will return a new DataFrame with a subset of the columns in the original. For example,

    .. ipython:: python

        new_df = df[["label", "img"]]
        new_df.columns

    Passing a ``str`` that isn't among the column names will raise a ``KeyError``.

```

:::{admonition} Copy vs. Reference
See {doc}`copying` for more information.

You may be wondering whether the columns returned by indexing are copies of the columns in the original DataFrame. The columns returned by the index operator reference the same columns in the original DataFrame. This means that modifying the columns returned by the index operator will modify the columns in the original DataFrame.
:::

## Selecting Rows

### Selecting Rows by Position

In Meerkat, the rows of a DataFrame or Column are ordered. This means that rows are
uniquely identified by their position in the DataFrame or Column (similar to how the
elements of a [Python List](https://www.w3schools.com/python/python_lists.asp) are
uniquely identified by their position in the list).

Row indices range from 0 to the number of rows in the DataFrame or Column minus one. To
see how many rows a DataFrame or a column has we can use `len()`. For example,

```{eval-rst}
.. ipython:: python

   len(df)
```

Above we mentioned how a DataFrame could be viewed as a dictionary mapping column names
to columns. Equivalently, it also may be helpful to think of a DataFrame as a list of
dictionaries mapping column names to values. The DataFrame interface supports both of these
views – under the hood, storage is organized so as to make both column and row accesses fast.

```{eval-rst}
.. card::


    **Selecting a Single Row from a DataFrame**: ``int`` -> :class:`Dict[str, Any]`
    ^^^^^^^^^^^^^^

    To select a single row from a DataFrame, we simply pass it's position to the index operator. For example,

    .. ipython:: python

        row = df[2]
        row

    Passing an ``int`` that is less than ``0`` or greater than ``len(df)`` will raise an ``IndexError``.

```

Notice how `row` contains a full [PIL Image](https://pillow.readthedocs.io/en/stable/reference/Image.html).
With thousands of images in the dataset, it wouldn't make sense to hold all the images in memory.
Instead, images are only loaded into memory at the moment they are selected.

:::{admonition} Lazy Selection
*What if we want to select a row without loading the image into memory?* Meerkat supports lazy selection through the `lz` indexer.

```{eval-rst}
.. ipython:: python

    row = df[2]
    row
```

Notice that instead of holding the image in memory, `row` holds a {class}`~meerkat.FileCell` object.
This object knows how to load the image into memory, but stops just short of doing so. Later on, when we want to access the image, we can use the :meth:~meerkat.FileCell.get\` method on the cell. For example,

```{eval-rst}
.. ipython:: python

    row["img"].get()

```

Lazy selection is critical for manipulating and managing DataFrames in Meerkat.
It is discussed in more detail in the guide on {doc}`deferred`.
:::

The same position-based indexing works for selecting a single cell from a Column.

```{eval-rst}
.. card::

    **Selecting a Single Cell from a Column**: ``int`` -> :class:`Any`
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    To select a single cell from a column, we pass it's position to the index operator. For example,

    .. ipython:: python

        col = df["label"]
        col[2]

    Passing an ``int`` that is less than ``0`` or greater than ``len(df["label"])`` will raise an ``IndexError``.

```

There are three different ways to select a subset of rows from a DataFrame: via `slice`, `Sequence[int]`, or `Sequence[bool]`.

```{eval-rst}
.. card::

    **Selecting Multiple Rows from a DataFrame**: ``slice`` -> :class:`~meerkat.DataFrame`
    ^^^^^^^^^^^^^^^

    To select a set of contiguous rows from a DataFrame, we can use an integer slice ``[start:end]``.
    The subset of rows will be returned as a new DataFrame.

    .. ipython:: python

        new_df = df[50:100]
        new_df

    We can also use integer slices to select a set of evenly spaced rows from a DataFrame ``[start:end:step]``. For example, below we select every tenth row from the first 100 rows in the DataFrame.

    .. ipython:: python

        new_df = df[0:100:10]
        new_df

```

```{eval-rst}
.. card::

    **Selecting Multiple Rows from a DataFrame**: ``Sequence[int]`` -> :class:`~meerkat.DataFrame`
    ^^^^^^^^^^^^^^

    To select multiple rows from a DataFrame we can also pass a list of ``int``.

    .. ipython:: python

        small_df = df[[0, 2, 5, 8, 17]]
        small_df

    Other valid sequences of ``int`` that can be used to index are:

    * ``Tuple[int]`` – a tuple of integers.
    * ``np.ndarray[np.integer]`` - a NumPy NDArray with `dtype` `np.integer`.
    * ``pd.Series[np.integer]`` - a Pandas Series with `dtype` `np.integer`.
    * ``torch.Tensor[torch.int64]`` - a PyTorch Tensor with `dtype` `torch.int`.
    * ``mk.Column`` - a Meerkat column who's cells are ``int``, ``np.integer``, or ``torch.int64``.

    This is useful when the rows are neither coontiguous nor evenly spaced (otherwise slice
    indexing, described above, is faster).

```

```{eval-rst}
.. card::

    **Selecting Multiple Rows from a DataFrame**: ``Sequence[bool]`` -> :class:`~meerkat.DataFrame`
    ^^^^^^^^^^^^^^

    To select multiple rows from a DataFrame we can also pass a list of ``bool`` the
    same length as the DataFrame. Below we select the first and last rows from
    the smaller DataFrame ``small_df`` that we selected in the panel above.

    .. ipython:: python

        small_df[[True, False, False, False, True]]


    Other valid sequences of ``bool`` that can be used to select  are:

    * ``Tuple[bool]`` – a tuple of bool.
    * ``np.ndarray[bool]`` - a NumPy NDArray with `dtype` `bool`.
    * ``pd.Series[bool]`` - a Pandas Series with `dtype` `bool`.
    * ``torch.Tensor[torch.bool]`` - a PyTorch Tensor with `dtype` `torch.bool`.
    * ``mk.Column`` - a Meerkat column who's cells are ``int``, ``bool``, or ``torch.bool``.

    This is very useful for quickly selecting a subset of rows that satisfy a predicate
    (like you might do with a ``WHERE`` clause in SQL).
    For example, say we want to select all rows that have a value of ``"parachute"`` in
    the ``"label"`` column. We could do this using the following code:

    .. ipython:: python
        :okwarning:

        small_df[small_df["label"] == "parachute"]



```

### Selecting Rows by Primary Key

:::{admonition} For Pandas Users
`.iloc` **and** `.loc`:
Pandas users are likely familiar with `.iloc` and `.loc` properties of DataFrames and Series.
These properties are used to select data by integer position and by label in the index, respectively.In Meerkat, DataFrames and Columns do **not** have a designated index object as do DataFrames and Series. In Meerkat, the primary way to select rows in Meerkat is by integer position or boolean mask, so there is no need for distinct `.iloc` and `loc` indexers.

**Indexing Cells**:
In Pandas, it's possible to select a cell directly from a DataFrame with a single index like `df.loc[2, "label"]`.
This is **not** supported in Meerkat. Instead you should chain the indexing operators together. For example,
`df["label"][2]`. In general, you should index the column first and then the row. Doing it in the reverse order
could be wasteful, since the other cells in the row would be loaded for no reason.
:::

:::{admonition} Copy vs. Reference
See {doc}`copying` for more information.

You may be wondering whether the rows returned by indexing are copies or references of the rows in the original DataFrame.
This depends on (1) which of the selection strategies above you use (`slice` vs. `Sequence[int]` vs. `Sequence[bool]`)  and (2) the column type (*e.g.* {class}`PandasSeriesColumn`, {class}`TensorColumn`).

In general, columns inherit the copying behavior of their underlying data structure.
For example, a {class}`TensorColumn` has the copying behavior of a NumPy array, as described in the [Numpy indexing documentation](https://numpy.org/doc/stable/reference/arrays.indexing.html).
See a more detailed discussion in {doc}`copying`.
:::
