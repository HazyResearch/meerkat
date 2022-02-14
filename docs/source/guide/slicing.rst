Slicing and Selecting Data
===========================

As discussed in the :doc:`data_structures`, there are two key data structures in Meerkat: the Column and the DataPanel. In this guide, we'll demonstrate how to access the data in these structures.

Throughout, we'll be using the following DataPanel, which holds the Imagenette dataset, a small subset of the original ImageNet. This DataPanel includes a column holding images, a column holding their labels, and a few others.

.. ipython:: python

   import meerkat as mk
   dp = mk.datasets.get("imagenette")
   dp


Selecting Columns
------------------
The columns in a DataPanel are uniquely identified by ``str`` names. The code
below displays the column names in the Imagenette datapanel we loaded above: 

.. ipython:: python

   dp.columns

Using these column names, we can pull out an individual column or a subset of them as a new
DataPanel. 

.. panels::
    :column: col-lg-12 p-2


    **Selecting a Single Column**: ``str`` -> :class:`~meerkat.AbstractColumn`
    ^^^^^^^^^^^^^^

    To select a single column, we simply pass it's name to the index operator. For example,

    .. ipython:: python

        col = dp["label"]
        col

    Passing a ``str`` that isn't among the column names will raise a ``KeyError``.  
    
It may be helpful to think of a DataPanel as a dictionary mapping column names to columns. 
Indeed, a DataPanel implements other parts of the ``dict`` interface including :meth:`~meerkat.DataPanel.keys()`, :meth:`~meerkat.DataPanel.values()`, and :meth:`~meerkat.DataPanel.items()`. Unlike with a dictionary, you can access a subset of a DataPanel's columns.

.. panels::
    :column: col-lg-12 p-2


    **Selecting Multiple Columns**: ``Sequence[str]`` -> :class:`~meerkat.DataPanel`
    ^^^^^^^^^^^^^^

    You can select multiple columns by passing a list or tuple of column names. Doing so will return a new DataPanel with a subset of the columns in the original. For example,

    .. ipython:: python

        new_dp = dp[["label", "img"]]
        new_dp.columns

    Passing a ``str`` that isn't among the column names will raise a ``KeyError``.  


.. admonition:: Copy vs. Reference

    See :doc:`copying` for more information.
    
    You may be wondering whether the columns returned by indexing are copies of the columns in the original DataPanel. The columns returned by the index operator reference the same columns in the original DataPanel. This means that modifying the columns returned by the index operator will modify the columns in the original DataPanel. 



Selecting Rows
---------------

In Meerkat, the rows of a DataPanel or Column are ordered. This means that rows are 
uniquely identified by their position in the DataPanel or Column (similar to how the 
elements of a `Python List <https://www.w3schools.com/python/python_lists.asp>`_ are 
uniquely identified by their position in the list).

Row indices range from 0 to the number of rows in the DataPanel or Column minus one. To
see how many rows a DataPanel or a column has we can use ``len()``. For example,

.. ipython:: python

   len(dp)


.. panels::
    :column: col-lg-12 p-2


    **Selecting a Single Row from a DataPanel**: ``int`` -> :class:`Dict[str, Any]`
    ^^^^^^^^^^^^^^

    To select a single row from a DataPanel, we simply pass it's position to the index operator. For example,

    .. ipython:: python

        row = dp[2]
        row

    Passing an ``int`` that is less than ``0`` or greater than ``len(dp)`` will raise an ``IndexError``.  


.. admonition:: For Pandas Users: ``.iloc`` and ``.loc`` 

    Pandas users are likely familiar with ``.iloc`` and ``.loc`` properties of DataFrames and Series.
    These properties are used to select data by integer position and by label in the index, respectively.

    In Meerkat, DataPanels and Columns do **not** have a designated index object as do DataFrames and Series.
    In meerkat, the primary way to select rows in Meerkat is by integer position or boolean mask, so there is no need for distinct ``.iloc`` and ``loc`` indexers. 

Above we mentioned how a DataPanel could be viewed as a dictionary mapping column names 
to columns. Equivalently, it also may be helpful to think of a DataPanel as a list of 
dictionaries mapping column names to values. The DataPanel interface supports both of these 
views – under the hood, storage is organized so as to make both column and row accesses
as fast as possible.
    

.. panels::
    :column: col-lg-12 p-2

    **Selecting a Single Cell from a Column**: ``int`` -> :class:`Any`
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    To select a single cell from a column, we pass it's position to the index operator. For example,

    .. ipython:: python

        col = dp["label"]
        col[2]

    Passing an ``int`` that is less than ``0`` or greater than ``len(dp["label"])`` will raise an ``IndexError``.  

.. admonition:: For Pandas Users: Indexing Cells

    In Pandas, it's possible to select a cell directly from a DataFrame with a single index like ``df.loc[2, "label"]``. 
    This is **not** supported in Meerkat. Instead you should chain the indexing operators together. For example,
    ``dp["label"][2]``. In general, you should index the column first and then the row. Doing it in the reverse order
    could be wasteful, since the other cells in the row would be loaded for no reason.  

There are a few different ways to select a subset of rows from a DataPanel. 
.. panels::
    :column: col-lg-12 p-2

    **Selecting Multiple Rows from a DataPanel**: ``slice`` -> :class:`~meerkat.DataPanel`
    ^^^^^^^^^^^^^^^

    To select a set of contiguous rows from a DataPanel, we can use an integer slice ``[start:end]``. 
    The subset of rows will be returned as a new DataPanel. 

    .. ipython:: python
     
        new_dp = dp[50:100]
        new_dp
    
    We can also use integer slices to select a set of evenly spaced rows from a DataPanel ``[start:end:step]``. For example, below we select everyt tenth row from the first 100 rows in the DataPanel.

    .. ipython:: python
     
        new_dp = dp[0:100:10]
        new_dp
    
Note that Python lists share this same slicing syntax. However, unlike a Python list, a DataPanel's rows can be selected in a few other ways.


.. panels::
    :column: col-lg-12 p-2

    **Selecting Multiple Rows from a DataPanel**: ``Sequence[int]`` -> :class:`~meerkat.DataPanel`
    ^^^^^^^^^^^^^^

    To select multiple rows from a DataPanel we can also pass a list of ``int``.

    .. ipython:: python
     
        new_dp = dp[[0, 2, 5, 8, 17]]
        new_dp

    Other valid sequences of ``int`` that can be used to index are:

    * ``Tuple[int]`` – a tuple of integers.
    * ``np.ndarray[np.integer]`` - a NumPy NDArray with `dtype` `np.integer`.
    * ``pd.Series[np.integer]`` - a Pandas Series with `dtype` `np.integer`.
    * ``torch.Tensor[torch.int64]`` - a PyTorch Tensor with `dtype` `torch.int`.
    * ``mk.AbstractColumn`` - a Meerkat column who's cells are ``int``, ``np.integer``, or ``torch.int64``.  

    This is useful when the rows are neither coontiguous nor evenly spaced (otherwise slice 
    indexing, described above, is faster).    


.. panels::
    :column: col-lg-12 p-2

    **Selecting Multiple Rows from a DataPanel**: ``Sequence[bool]`` -> :class:`~meerkat.DataPanel`
    ^^^^^^^^^^^^^^

    To select multiple rows from a DataPanel we can also pass a list of ``bool`` the 
    same length as the DataPanel. Below we select the first and last rows from 
    the smaller DataPanel ``new_dp`` that we selected in the panel above. 

    .. ipython:: python

        new_dp[[True, False, False, False, True]]
        

    Other valid sequences of ``bool`` that can be used to select  are:
    
    * ``Tuple[bool]`` – a tuple of bool.
    * ``np.ndarray[bool]`` - a NumPy NDArray with `dtype` `bool`.
    * ``pd.Series[bool]`` - a Pandas Series with `dtype` `bool`.
    * ``torch.Tensor[torch.bool]`` - a PyTorch Tensor with `dtype` `torch.bool`.
    * ``mk.AbstractColumn`` - a Meerkat column who's cells are ``int``, ``bool``, or ``torch.bool``.  

    This is very useful for quickly filtering DataPanels. 



.. admonition:: Copy vs. Reference

    See :doc:`copying` for more information.
    
    You may be wondering whether the rows returned by indexing are copies of the rows in the original DataPanel. 
    This depends on (1) which of the selection strategies above you use (``slice`` vs. ``Sequence[int]`` vs. ``Sequence[bool]``)  and (2) the column type (**e.g.** PandasSeriesColumn)




