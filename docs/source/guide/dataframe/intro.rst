
Introduction to DataFrames 
================================

Meerkat provides two data structures, the column and the dataframe, that together help 
you build, manage, and explore machine learning datasets . Everything you do with Meerkat will 
involve one or both of these data structures, so we begin this user guide with their
high-level introduction. 

Column
-------
A column is a sequential data structure (analagous to a `Series <https://pandas.pydata.org/docs/reference/api/pandas.Series.html>`_ in Pandas or a `Vector <https://cran.r-project.org/doc/manuals/r-release/R-intro.html#Simple-manipulations-numbers-and-vectors>`_ in R). 
Meerkat supports a diverse set of column types (*e.g.* :class:`~meerkat.TensorColumn`, 
:class:`~meerkat.ImageColumn`), each intended for different kinds of data. To see a
list of the core column types and their capabilities, see :doc:`column_types`.

Below we create a simple column to hold a set of images stored on disk. To create it,
we simply pass filepaths to the :class:`~meerkat.ImageColumn` constructor.

.. ipython:: python

    @suppress
    import os
    import meerkat as mk
    @suppress
    abs_path_to_img_dir = os.path.join(os.path.dirname(os.path.dirname(mk.__file__)), "docs/assets/guide/data_structures")
    img_col = mk.ImageColumn(
        ["img_0.jpg", "img_1.jpg", "img_2.jpg"], 
        base_dir=abs_path_to_img_dir
    )
    img_col

    @suppress
    from display import display_df 
    @suppress
    display_df(img_col, "simple_column")

.. raw:: html
   :file: ../html/display/simple_column.html

All Meerkat columns are subclasses of :class:`~meerkat.Column` and share a common 
interface, which includes :meth:`~meerkat.Column.__len__`, :meth:`~meerkat.Column.__getitem__`, :meth:`~meerkat.Column.__setitem__`, :meth:`~meerkat.Column.filter`, :meth:`~meerkat.Column.map`, and :meth:`~meerkat.Column.concat`. Below we get the length of the column we just created. 

.. ipython:: python

    len(img_col)


Certain column types may expose additional functionality. For example, 
:class:`~meerkat.TensorColumn` inherits most of the functionality of an
`ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_.

.. ipython:: python

    id_col = mk.TensorColumn([0, 1, 2])
    id_col.sum()
    id_col == 1

To see the full list of methods available to a column type, 

If you don't know which column type to use, you can just pass a familiar data 
structure like a ``list``, ``np.ndarray``, ``pd.Series``, and ``torch.Tensor`` to 
:meth:`~meerkat.Column.from_data` and Meerkat will automatically pick an
appropriate column type. 

.. ipython:: python

    import torch
    tensor = torch.tensor([1,2,3])
    mk.Column.from_data(tensor)

DataFrame
----------
A :class:`DataFrame` is a collection of equal-length columns (analagous to a `DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame>`_ in Pandas or R). 
DataFrames in Meerkat are used to manage datasets and per-example artifacts (*e.g.* model predictions and embeddings).  

Below we combine the columns we created above into a single DataFrame. We also add an 
additional column containing labels for the images. Note that we can pass non-Meerkat data 
structures like ``list``, ``np.ndarray``, ``pd.Series``, and ``torch.Tensor``  directly to the 
DataFrame constructor and Meerkat will infer the column type. We do not need to first 
convert to a Meerkat column. 

.. ipython:: python

    df = mk.DataFrame(
        {
            "img": img_col,
            "label": ["boombox", "truck", "dog"],
            "id": id_col, 
        }
    )
    df 

    @suppress
    from display import display_df 
    @suppress
    display_df(df, "simple_df")

.. raw:: html
   :file: ../html/display/simple_df.html

Read on to learn how we access the data in Columns and DataFrames.