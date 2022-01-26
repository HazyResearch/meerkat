Slicing and Selecting Data
===========================

As discussed in the :doc:`data_structures`, there are two key data structures in Meerkat: the Column and the DataPanel. In this guide, we will show how to access the data in these structures.

Throughout, we'll be accessing data from the following DataPanel holding the Imagenette dataset, a small subset of the original ImageNet. This DataPanel includes a column holding images and a column holding their labels, among a few other columns.

.. ipython:: python

   import meerkat as mk
   dp = mk.datasets.get("imagenette")
   dp


Selecting Columns
---------------
The columns in a DataPanel are uniquely identified by ``str`` names. For example, the code
below displays the column names in the Imagenette datapanel we loaded above: 

.. ipython:: python

   dp.columns

Using these column names, we can pull out an individual column or a subset of them as a new
DataPanel. 

.. panels::
    :column: col-lg-12 p-2


    **Selecting a Single Column** ``str`` -> :class:`~meerkat.AbstractColumn`
    ^^^^^^^^^^^^^^

    To select a single column, we simply pass it's name to the index operator. For example,

    .. ipython:: python

        col = dp["label"]
        col

    Passing a ``str`` that isn't among the column names will raise a ``KeyError``.  
    
It may be helpful to think of a DataPanel as a dictionary mapping column names to columns. 
Indeed, a DataPanel implements other parts of the ``dict`` interface including :meth:`~meerkat.DataPanel.keys()`, :meth:`~meerkat.DataPanel.values()`, and :meth:`~meerkat.DataPanel.items()`. 

.. panels::
    :column: col-lg-12 p-2


    **Selecting Multiple Columns** ``List[str] | Tuple[str]`` -> :class:`~meerkat.DataPanel`
    ^^^^^^^^^^^^^^

    You can also select multiple columns by passing a list or tuple of column names. Doing so will return a new DataPanel with a subset of the columns in the original. For example,

    .. ipython:: python

        new_dp = dp[["label", "img"]]
        new_dp.columns

    Passing a ``str`` that isn't among the column names will raise a ``KeyError``.  


.. admonition:: Copy vs. Reference

    See :doc:`copying` for more information.
    
    You may be wondering whether the columns returned by indexing are copies of the columns in the original DataPanel. The columns returned by the index operator reference the same columns in the original DataPanel. This means that modifying the columns returned by the index operator will modify the columns in the original DataPanel. 



Row Indexing 
------------


.. note::
    Pandas users are likely familiar with ``.iloc`` and ``.loc`` properties of DataFrames and Series.
    These properties are used to select data by integer position and by label in the index, respectively.
    In Meerkat, DataPanels and Columns do __not__ have a designated index object as do DataFrames and Series in Pandas.
    The primary way to select data in Meerkat is by integer position or boolean mask, so there is no need for 
    distinct ``.iloc`` and ``loc`` indexers. 




